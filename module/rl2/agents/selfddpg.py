from __future__ import division
from collections import deque
import os
import warnings

import numpy as np
import keras2.backend as K
import keras2.optimizers as optimizers

from ..selfcore import self_Agent, sample_Agent
from rl2.random import OrnsteinUhlenbeckProcess
from rl2.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


def _all_weights(layers):
    out = []
    for layer in layers:
        out.append(layer.get_weights())
    return out

def _NN_params(actor):
    params = []
    for layer in actor.layers:
        if len(layer.get_weights())==0:
            continue
        else:
            w, b = layer.get_weights()
            layer_params = np.hstack((w.flatten(), b.flatten()))
            params = np.hstack((params, layer_params))
    params = np.array(params).flatten()
    return params


def push_out(arr, insert_object):
    arr.append(insert_object)
    del arr[0]
    return arr

def next_state_gradient(state0, action, tau, hypara):
    # linear system
    m, l, g = hypara
    A = np.array([[0, 1], [(3*g)/(2*l), 0]])
    B = np.array([0, 3/(m*l**2)])
    
    # ∂s'/∂τ
    dsdt = np.dot(np.dot(A, _array_exp(A*tau)), state0) + B*action

    # ∂s'/∂u
    dsdu = np.dot(np.dot(_array_exp(A*tau), np.linalg.inv(A)), B) \
         - np.dot(np.dot(_array_exp(A*tau), np.linalg.inv(A)), np.dot(_array_exp(-A*tau), B))

    return [dsdt, dsdu]

def _array_exp(A):
    v, p = np.linalg.eig(A)
    align = np.array([[v[0], 0],[0, v[1]]])
    exp = np.exp(align)
    exp[~np.eye(exp.shape[0], dtype=bool)] = 0
    out = np.dot(np.dot(p, exp), np.linalg.inv(p))
    return out


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class selfDDPGAgent(self_Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory, action_clipper=[-10.,10.], tau_clipper=[0.01, 1.],
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, coef_u=1., coef_tau=0.01,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, mb_noise=False, custom_model_objects={}, target_model_update=.001, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(selfDDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        if random_process == 'OrnsteinUhlenbeckProcess':
            random_process = OrnsteinUhlenbeckProcess(1., size=3)
        elif random_process is None:
            pass
        else:
            assert False, 'Unknown Random Process'

        # Parameters.
        self.action_clipper = action_clipper
        self.tau_clipper = tau_clipper
        self.outputlayer_param_mean = []
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.mb_noise = mb_noise
        self.coef_u = coef_u
        self.coef_tau = coef_tau
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory

        # ibuki_made
        arr = []
        for _ in range(500):
            tmp = _all_weights(self.actor.layers)
            arr.append(tmp)
        self.agents_log = arr  # とりあえずなんでもいい

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and never use it.
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        state_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                state_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(state_inputs)

        combined_output = self.critic(combined_inputs)
        self.combined_output, self.state_inputs = combined_output, state_inputs

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output))
        # loss = -K.mean()になっているのは, policy_gradientを平均値で近似するため
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        self.actor_updates = updates
        self.state_inputs = state_inputs
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(state_inputs + [K.learning_phase()],
                                             [self.actor(state_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                state_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(state_inputs, [self.actor(state_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def _add_mb_noise(self, state, actor_output, c_u=0.1, c_tau=0.2):
        # 次の状態の行動に対する勾配によってノイズスケールを決める
        state = state[0]
        hypara = [1,1,10]
        next_state_action_gradient, next_state_tau_gradient = next_state_gradient(state, actor_output[0], actor_output[1], hypara)
        size_of_gu = np.linalg.norm(next_state_action_gradient)
        size_of_gt = np.linalg.norm(next_state_tau_gradient)

        coef_u = c_u / (size_of_gu + c_u)
        coef_tau = c_tau / (size_of_gt + c_tau)

        action, tau = actor_output
        action += np.random.randn() * coef_u
        tau += np.random.randn() * coef_tau
        return np.array([action, tau])


    def _add_gaussian(self, actor_output, coef_u, coef_tau):
        action, tau = actor_output
        action += np.random.randn() * coef_u
        tau += np.random.randn() * coef_tau
        return np.array([action, tau])


    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        if self.training:
            if self.mb_noise:
                action = self._add_mb_noise(state, action)
            else:
                action = self._add_gaussian(action, self.coef_u, self.coef_tau)
        
        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        #TODO: change the law of selecting action
        action = self.select_action(state)
        #clip
        action = action.clip(min=np.array([self.action_clipper[0], self.tau_clipper[0]]),\
                             max=np.array([self.action_clipper[1], self.tau_clipper[1]]))

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action #ここじゃなくてCBFの後に入れるべき

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            # Make a mini-batch to learn. So batch learning is done in every time steps.
            #This is based on the paper.
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = [] # current_state
            reward_batch = [] # current_reward
            action_batch = [] # current_action
            terminal1_batch = []
            state1_batch = [] # next_state
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            # Change batch class from list to np.ndarray
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions), (action_batch.shape, (self.batch_size, self.nb_actions))

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                # state0_batch_with_action is input, targets is teacher
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2: # state
                    inputs = state0_batch[:]
                else: #now we get len = 1
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                self.inputs = inputs
                action_values = self.actor_train_fn(inputs)[0] # actor update with critics loss
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()
        
        return metrics

    def agent_copy_from_layers(self, weights):
        for i, layer in enumerate(weights):
            self.actor.layers[i].set_weights(layer)





class selfDDPGAgent2(selfDDPGAgent):
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory, action_clipper=[-10., 10.], tau_clipper=[0.01, 1.],
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, coef_u=1., coef_tau=.01,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf, params_logging=False, gradient_logging=False,
                 random_process=None, mb_noise=False, custom_model_objects={}, target_model_update=.001, **kwargs):
        super().__init__(nb_actions=nb_actions, actor=actor, critic=critic,
                 critic_action_input=critic_action_input, memory=memory, action_clipper=action_clipper, tau_clipper=tau_clipper,
                 gamma=gamma, batch_size=batch_size, nb_steps_warmup_critic=nb_steps_warmup_critic,
                 coef_u=coef_u, coef_tau=coef_tau,
                 nb_steps_warmup_actor=nb_steps_warmup_actor,
                 train_interval=train_interval, memory_interval=memory_interval, delta_range=delta_range,
                 delta_clip=delta_clip,
                 random_process=random_process,
                 mb_noise=mb_noise,
                 custom_model_objects=custom_model_objects,
                 target_model_update=target_model_update)
        self.gradient_log = []
        self.gradient_logging = gradient_logging
        self.params_logging = params_logging
        
    def compile(self, optimizer, metrics=[], action_lr=0.001, tau_lr=0.00001):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and never use it.
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        state_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                state_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(state_inputs)
        combined_output = self.critic(combined_inputs) # Q(s,a|θ^Q)を示してる

        # 学習の進捗を確認するために, 全パラメータに対するgradientを確認する
        # combined_outputはQ(s,a|θ^Q) (critic) の出力そのもので,
        # K.mean(combined_output)にマイナスをつけることで, 最小化がQ関数の最大化と同じ役割をはたす.
        # lossを-K.mean(combined_output)にすることでgradientはそれを小さくする方向に動く.
        # meanなのは, memoryの分布がrho^piに従ってるという仮定でやってる
        dummy_optimizer = optimizers.Optimizer()
        self.combined_inputs = combined_inputs
        self.loss_func = -K.mean(combined_output)
        self.gradients = dummy_optimizer.get_gradients(-K.mean(combined_output), self.actor.trainable_weights)
        self.gradient_calculate_function = self._gradient_calculate_function()

        # action_params_update と tau_params_updateで分ける
        action_tw, tau_tw = self._split_params()
        # action params update
        # ∂Q/∂a
        action_updates = _get_updates_original(action_tw, -K.mean(combined_output), action_lr)
        # tau params update
        # ∂Q/∂τ
        tau_updates = _get_updates_original(tau_tw, -K.mean(combined_output), tau_lr)

        updates = action_updates + tau_updates
        if self.target_model_update < 1.:
            # Include soft target model updates.
            # target network update operations
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for Batch Normalization

        # Finally, combine it all into a callable function.
        self.actor_updates = updates
        self.state_inputs = state_inputs
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(state_inputs + [K.learning_phase()],
                                             [self.actor(state_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                state_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(state_inputs, [self.actor(state_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def _split_params(self):
        tw = self.actor.trainable_weights
        action_tw = []
        tau_tw = []
        for i in range(len(tw)):
            if i % 4 < 2:
                action_tw.append(tw[i])
            else:
                tau_tw.append(tw[i])
        return action_tw, tau_tw
        
    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            # Make a mini-batch to learn. So batch learning is done in every time steps.
            #This is based on the paper.
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = [] # current_state
            reward_batch = [] # current_reward
            action_batch = [] # current_action
            terminal1_batch = []
            state1_batch = [] # next_state
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            # Change batch class from list to np.ndarray
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions), (action_batch.shape, (self.batch_size, self.nb_actions))

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                # state0_batch_with_action is input, targets is teacher
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2: # state
                    inputs = state0_batch[:]
                else: #now we get len = 1
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                self.inputs = inputs
                if self.gradient_logging:
                    current_gradient = self._get_current_gradient(inputs)
                    norm = gradient_evaluation(current_gradient)
                    self.gradient_log.append(norm)
                action_values = self.actor_train_fn(inputs)[0] # actor update with critics loss
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()
        
        if self.params_logging:
            self.params_log.append(_NN_params(self.actor))
        return metrics

    def _gradient_calculate_function(self):
        # function input
        input_tensors = [self.combined_inputs[1]]
        gradient_tensors = self.gradients

        # tensorflow running function
        gradient_calculate_function = K.function(input_tensors, gradient_tensors)
        return gradient_calculate_function

    def _get_current_gradient(self, state0_batch):
        gradient_calculate_function = self.gradient_calculate_function

        # value
        gradient_values = gradient_calculate_function(state0_batch)
        return gradient_values


class selfDDPGAgent3(selfDDPGAgent):
    """Adaptive discount factor to approximate exp(- alpha * tau).
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory, action_clipper=[-10., 10.], tau_clipper=[0.01, 10.],
                 alpha=.4, gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, coef_u=1., coef_tau=.01,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, mb_noise=False, custom_model_objects={}, target_model_update=.001, **kwargs):
        super().__init__(nb_actions=nb_actions, actor=actor, critic=critic,
                 critic_action_input=critic_action_input, memory=memory, action_clipper=action_clipper, tau_clipper=tau_clipper,
                 gamma=gamma, batch_size=batch_size, nb_steps_warmup_critic=nb_steps_warmup_critic,
                 coef_u=coef_u, coef_tau=coef_tau,
                 nb_steps_warmup_actor=nb_steps_warmup_actor,
                 train_interval=train_interval, memory_interval=memory_interval, delta_range=delta_range,
                 delta_clip=delta_clip,
                 random_process=random_process,
                 mb_noise=mb_noise,
                 custom_model_objects=custom_model_objects,
                 target_model_update=target_model_update)
        self.alpha = alpha
        
    def compile(self, optimizer, metrics=[], action_lr=0.001, tau_lr=0.00001):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and never use it.
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        state_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                state_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(state_inputs)
        combined_output = self.critic(combined_inputs) # Q(s,a|θ^Q)を示してる

        self.combined_inputs = combined_inputs

        # action_params_update と tau_params_updateで分ける
        action_tw, tau_tw = self._split_params()
        # action params update
        # ∂Q/∂a
        action_updates = _get_updates_original(action_tw, -K.mean(combined_output), action_lr)
        # tau params update
        # ∂Q/∂τ
        tau_updates = _get_updates_original(tau_tw, -K.mean(combined_output), tau_lr)

        updates = action_updates + tau_updates
        if self.target_model_update < 1.:
            # Include soft target model updates.
            # target network update operations
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for Batch Normalization

        # Finally, combine it all into a callable function.
        self.actor_updates = updates
        self.state_inputs = state_inputs
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(state_inputs + [K.learning_phase()],
                                             [self.actor(state_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                state_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(state_inputs, [self.actor(state_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def _split_params(self):
        tw = self.actor.trainable_weights
        action_tw = []
        tau_tw = []
        for i in range(len(tw)):
            if i % 4 < 2:
                action_tw.append(tw[i])
            else:
                tau_tw.append(tw[i])
        return action_tw, tau_tw
        
    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            # Make a mini-batch to learn. So batch learning is done in every time steps.
            #This is based on the paper.
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = [] # current_state
            reward_batch = [] # current_reward
            action_batch = [] # current_action
            terminal1_batch = []
            state1_batch = [] # next_state
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            # Change batch class from list to np.ndarray
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions), (action_batch.shape, (self.batch_size, self.nb_actions))

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                # gamma should be exp(- alpha * tau)
                tau_mean = np.mean(self.actor.predict_on_batch(state1_batch)[:,1])
                gamma = np.exp(- self.alpha * tau_mean)
                discounted_reward_batch = gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                # state0_batch_with_action is input, targets is teacher
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2: # state
                    inputs = state0_batch[:]
                else: #now we get len = 1
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                self.inputs = inputs
                action_values = self.actor_train_fn(inputs)[0] # actor update with critics loss
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()
        
        return metrics


def _get_updates_original(params, loss, lr):
    optimizer = optimizers.Adam(lr=lr, clipnorm = 1.)
    updates = optimizer.get_updates(loss=loss, params=params)
    return updates

def gradient_evaluation(gradient_values):
    if len(gradient_values[0].shape) == 2:
        g_array = gradient_values[0].flatten()
    else:
        g_array = gradient_values[0]
    for g in gradient_values[1:]:
        if len(g.shape) == 2:
            g = g.flatten()
        g_array = np.hstack((g_array, g))
    gradient_eval = np.linalg.norm(g_array)
    return gradient_eval


class sampleDDPGAgent(sample_Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory, action_clipper=10.,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(sampleDDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        if random_process == 'OrnsteinUhlenbeckProcess':
            random_process = OrnsteinUhlenbeckProcess(1., size=3)
        elif random_process is None:
            pass
        else:
            assert False, 'Unknown Random Process'

        # Parameters.
        self.action_clipper = action_clipper
        self.outputlayer_param_mean = []
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and never use it.
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        state_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                state_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(state_inputs)

        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output))
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(state_inputs + [K.learning_phase()],
                                             [self.actor(state_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                state_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(state_inputs, [self.actor(state_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        #TODO: change the law of selecting action
        action = self.select_action(state)
        action = np.clip(action, -self.action_clipper, self.action_clipper)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            # Make a mini-batch to learn. So batch learning is done in every time steps.
            #This is based on the paper.
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = [] # current_state
            reward_batch = [] # current_reward
            action_batch = [] # current_action
            terminal1_batch = []
            state1_batch = [] # next_state
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            # Change batch class from list to np.ndarray
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,), f'{reward_batch.shape}, {(self.batch_size,)}'
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions), (action_batch.shape, (self.batch_size, self.nb_actions))

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                # state0_batch_with_action is input, targets is teacher
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            tmp = self.actor.layers[4].get_weights()[0]
            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2: # state
                    inputs = state0_batch[:]
                else: #now we get len = 1
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0] # update actor here with critics loss
                # where is policy gradient?
                assert action_values.shape == (self.batch_size, self.nb_actions)
            actor_outputlayer_diff = tmp - self.actor.layers[4].get_weights()[0]
            diff_mean = np.mean(np.abs(actor_outputlayer_diff), axis=0)
            self.outputlayer_param_mean.append(diff_mean)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
