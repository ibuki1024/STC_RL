import numpy as np
import scipy

def moving_average(data, l=30):
    out = []
    if type(data) == list:
        data = np.array(data)
    for i in range(l, data.shape[0]):
        ma = np.mean(data[i-l:i])
        out.append(ma)
    out = np.array(out)
    return out


def lqr(A, B, Q, R):
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R).dot(B.T).dot(P)
    return -K


def dlqr(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(np.dot(B.T, P).dot(B) + R).dot(B.T).dot(P).dot(A)
    return -K


def discretized_system(A, B, dt):
    Ad = np.eye(A.shape[0]) + dt * A
    Bd = dt * B
    return Ad, Bd

def _gain(env, dt=None):
    m, l, g = env.m, env.l, env.g

    A = np.array([[0, 1], [(3*g)/(2*l), 0]])
    B = np.array([[0], [3/(m*l**2)]])
    Q = np.array([[1, 0], [0, 0.1]])
    R = np.array([[0.001]])
    
    if dt is not None:
        Ad, Bd = discretized_system(A, B, dt)  
        K = dlqr(Ad,Bd,Q,R)[0]
    else:
        K = lqr(A,B,Q,R)[0]
    
    return K


def make_standup_agent(actor_net, tau, env, epochs=100, verbose=False):
    # 学習データの用意
    action_repetition = int(np.ceil(20 * tau))  # minimum natural number which makes `dt` smaller than 0.005
    dt = tau / action_repetition
    K = _gain(env, dt)
    x_train = []
    y_train = []
    high = np.array([np.pi, np.pi])
    for i in range(30000):
        x_train.append([np.random.uniform(low=-high, high=high)])
        y_train.append([np.dot(K, x_train[-1][0]), tau])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 学習
    actor_net.compile(loss='mean_squared_error',optimizer='adam')
    actor_net.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=verbose)

    return actor_net

def get_NN_params(actor):
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


def set_NN_params(actor, params):
    param_idx = 0
    for layer in actor.layers:
        if len(layer.get_weights())==0:
            continue
        else:
            w, b = layer.get_weights()
            # set w
            w_prime = params[param_idx:param_idx+w.flatten().shape[0]].reshape(w.shape)
            param_idx += w.flatten().shape[0]

            # set b
            b_prime = params[param_idx:param_idx+b.flatten().shape[0]].reshape(b.shape)
            param_idx += b.flatten().shape[0]

            layer.set_weights([w_prime, b_prime])
    assert params.shape[0] == param_idx
