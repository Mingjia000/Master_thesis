import numpy as np
import par
function=par.function #0 linear, 1 pisewise, 2 nonlinear
def shipper_reward(route, shipper_h):
    # can be randomly distributed
    # s is the same for the same shipper?
    # the parameters
    # cost, time, delay,emission, transhipment
    '''
    reward = np.array([0 for i in range(len(route[:, 0]))])
    c1 = np.percentile(route[:, 0], 30)
    c2 = np.percentile(route[:, 0], 60)
    theta2 = 3 / 2 * np.log(c1) ** 2
    theta3 = 3 * np.log(c1) * np.log(c2)
    gamma2 = -0.5 * np.log(c1) ** 3
    gamma3 = -0.5 * np.log(c1) * (3 * np.log(c2) ** 2 + np.log(c1) ** 2)
    reward = reward.astype(float)

    for i in range(len(route[:, 0])):

        if route[i, 0] < c1:
            cost = np.log(route[i, 0]) ** 3
            reward[i] = -0.00759 * cost - 0.024 * route[i, 1]  # -0.0702*route[i,2]
        elif c1 < route[i, 0] < c2:
            cost = theta2 * np.log(route[i, 0]) ** 2 + gamma2
            reward[i] = -0.00759 * cost - 0.024 * route[i, 1]  # -0.0702*route[i,2]
        else:
            cost = theta3 * np.log(route[i, 0]) + gamma3
            reward[i] = -0.00759 * cost - 0.024 * route[i, 1]  # -0.0702*route[i,2]
    '''

    reward = np.zeros(len(route[:, 0]))
    mu, beta = 0, 1
    s = np.random.gumbel(mu, beta, len(route[:, 0]))

    reward = reward.astype(float)

    coeff=np.array([[-10,-8*5,-5*5,-2*5,-2],
                    [-10*5,-8,-5,-2,-2],
                    [-10,-8*2.5,-5*2.5,-2*5,-2*5],
                    [-10*5,-8*5,-5*5,-2,-2*5]])
    '''coeff = np.array([[-10, 0, 0, -2 * 5, -2],
                      [0, -8, 0, -2, -2],
                      [-10, -8 * 2.5, 0, -2 * 5, -2 * 5],
                      [-10 * 5, 0, -5 * 5, 0, -2 * 5]])'''

    if function == 2:
        c1 = np.percentile(route[:, 0], 30)
        c2 = np.percentile(route[:, 0], 60)
        theta2 = 3 / 2 * np.log(c1) ** 2
        theta3 = 3 * np.log(c1) * np.log(c2)
        gamma2 = -0.5 * np.log(c1) ** 3
        gamma3 = -0.5 * np.log(c1) * (3 * np.log(c2) ** 2 + np.log(c1) ** 2)
        reward = reward.astype(float)

        for i in range(len(route[:, 0])):
            h=int(shipper_h[i])
            if route[i, 0] < c1:
                cost = np.log(route[i, 0]) ** 3
            elif c1 < route[i, 0] < c2:
                cost = theta2 * np.log(route[i, 0]) ** 2 + gamma2
            else:
                cost = theta3 * np.log(route[i, 0]) + gamma3
            reward[i] = coeff[h, 0] * cost + np.sum(coeff[h, 1:] * route[i, 1:])

    if function == 0:
        for i in range(len(route[:, 0])):
            h = int(shipper_h[i])
            reward[i] = np.sum(coeff[h, :] * route[i, :])
            #reward[i] =  - 8 * 5 * route[i, 1] - 5 * 5 * route[i, 2]
        # cost: Nicolet;Masoud Khakdaman2020
        # VOT: Tao page221,Kollol Shams

    if function == 1:
        c = np.percentile(route[:, 0], 50)
        t = np.percentile(route[:, 1], 50)
        d = np.percentile(route[:, 2], 50)
        e = np.percentile(route[:, 3], 50)
        tr = np.percentile(route[:, 4], 50)
        r0 = route[:, 0] - c
        r1 = route[:, 1] - t
        r2 = route[:, 2] - d
        r3 = route[:, 3] - e
        r4 = route[:, 4] - tr
        r0 [r0<0] = 0
        r1 [r1<0] = 0
        r2 [r2<0] = 0
        r3 [r3<0] = 0
        r4 [r4<0] = 0
        for i in range(len(route[:, 0])):
            h = int(shipper_h[i])
            reward[i] = np.sum(coeff[h, :] * route[i, :]) + \
                        0.5*(coeff[h, 0]  * r0[i] + \
                        coeff[h, 1] * r1[i] + \
                        coeff[h, 2] * r2[i] + \
                        coeff[h, 3] * r3[i] + \
                        coeff[h, 4] * r4[i])

    reward = reward + s
    return reward

def shipper_compare(reward_0, reward_1):
    compare = np.array([0 for i in range(len(reward_1))])
    for i in range(len(reward_1)):
        if reward_0[i] > reward_1[i]:
                compare[i] = 0
        elif reward_0[i] < reward_1[i]:
                compare[i] = 1
        else:
                compare[i] = 0.5
    return compare



