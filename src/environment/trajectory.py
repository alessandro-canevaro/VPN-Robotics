import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from src.environment.gridworld import MRS

def compute_center_path(old_mrs, new_mrs, theta_i, theta_f, block_size=0.3, k=0.2):
    xi, yi = old_mrs.pos * block_size
    xf, yf = new_mrs.pos * block_size
    #print(xi, yi, np.rad2deg(theta_i), xf, yf, np.rad2deg(theta_f))

    x = lambda s: s**3*xf - (s-1)**3*xi + (k*np.cos(theta_f)-3*xf)*s**2*(s-1) + (k*np.cos(theta_i)+3*xi)*s*(s-1)**2
    y = lambda s: s**3*yf - (s-1)**3*yi + (k*np.sin(theta_f)-3*yf)*s**2*(s-1) + (k*np.sin(theta_i)+3*yi)*s*(s-1)**2
    return x, y

def compute_robots_path(old_mrs, new_mrs, theta_i, theta_f, block_size=0.3, k=0.2):
    x, y = compute_center_path(old_mrs, new_mrs, theta_i, theta_f, block_size, k)

    xi, yi = old_mrs.pos * block_size
    mi = np.tan(np.deg2rad(old_mrs.angle))
    xf, yf = new_mrs.pos * block_size
    mf = np.tan(np.deg2rad(new_mrs.angle))
    #print("mf", mf)

    try:
        rot_center = np.linalg.solve(np.array([[-mi, 1],
                                            [-mf, 1]]),
                                    np.array([yi-mi*xi, yf-mf*xf]))
        m = lambda s: (y(s)-rot_center[1])/(x(s)-rot_center[0])
        theta = lambda s: np.concatenate(([0.0], np.sign(new_mrs.angle-old_mrs.angle)*np.rad2deg(np.arctan(np.absolute((mi-m(s[1:]))/(1+mi*m(s[1:])))))))
    except np.linalg.LinAlgError:
        theta = lambda s: 0.0
        rot_center = np.array([0, 0])

    pos = old_mrs.get_agents_positions()*block_size
    r1, r2 = pos[0], pos[1]
    cond1 = abs(np.linalg.norm(r1-rot_center, ord=2) + np.linalg.norm(r2-rot_center, ord=2) - np.linalg.norm(r1-r2, ord=2)) < 0.01

    pos = new_mrs.get_agents_positions()*block_size
    r1, r2 = pos[0], pos[1]
    cond2 = abs(np.linalg.norm(r1-rot_center, ord=2) + np.linalg.norm(r2-rot_center, ord=2) - np.linalg.norm(r1-r2, ord=2)) < 0.01
    if cond1 and cond2:
        xa = lambda s, rl, ra: x(s) + block_size * rl * np.cos(np.deg2rad(old_mrs.angle+ra) + s*(np.deg2rad(new_mrs.angle+ra)-np.deg2rad(old_mrs.angle+ra)))
        ya = lambda s, rl, ra: y(s) + block_size * rl * np.sin(np.deg2rad(old_mrs.angle+ra) + s*(np.deg2rad(new_mrs.angle+ra)-np.deg2rad(old_mrs.angle+ra)))
    else:
        theta_a = lambda s, ra: np.deg2rad(old_mrs.angle+ra+theta(s))
        xa = lambda s, rl, ra: x(s) + np.cos(theta_a(s, ra)) * rl * block_size
        ya = lambda s, rl, ra: y(s) + np.sin(theta_a(s, ra)) * rl * block_size
    return xa, ya

def compute_reference_input(x, y, ds):
    x_dot, y_dot = np.gradient(x, ds), np.gradient(y, ds)
    x_dotdot, y_dotdot = np.gradient(x_dot, ds), np.gradient(y_dot, ds)
    theta = np.arctan2(y_dot, x_dot)
    v = np.sqrt(x_dot**2 + y_dot**2)
    w = (np.multiply(y_dotdot, x_dot) - np.multiply(x_dotdot, y_dot))/(x_dot**2 + y_dot**2)
    #w[0], w[1], w[-1], w[-2] = w[2], w[2], w[-3], w[-3]
    return theta, v, w

class OdometryModel:
    def __init__(self, x0, y0, theta0, noise) -> None:
        self.noise = noise
        self.x = x0
        self.y = y0
        self.theta = theta0

    def update(self, v, w, dt):
        self.x = self.x + dt * v * np.cos(self.theta) + dt*np.random.normal(0.0, self.noise)
        self.y = self.y + dt * v * np.sin(self.theta) + dt*np.random.normal(0, self.noise)
        self.theta = self.theta + dt * w + dt*np.random.normal(0, self.noise)

    def get_state(self):
        return self.x, self.y, self.theta


def _angle_diff(a, b):
    a = np.arctan2(np.sin(a), np.cos(a))
    b = np.arctan2(np.sin(b), np.cos(b))
    d1 = a - b
    d2 = 2 * np.pi - np.abs(d1)
    if d1 > 0:
        d2 *= -1.0
    if np.abs(d1) < np.abs(d2):
        return d1
    else:
        return d2

def nonlin_controller(x, y, theta, xr, yr, thetar, vr, wr):
    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    #theta, thetar = theta % (np.pi*2), thetar % (np.pi*2)
    #delta_theta = thetar - theta
    #delta_theta = delta_theta if abs(delta_theta) <= np.pi else (np.sign(delta_theta)*(np.pi*2) - delta_theta)
    delta_theta = _angle_diff(thetar, theta)
        
    e = R @ np.array([xr - x, yr - y, delta_theta]).T
    e1, e2, e3 = e[0], e[1], e[2]
    c, a = 0.7, 1
    k1 = 2*c*a
    u1 = -k1 * e1
    v = vr * np.cos(e3) - u1

    k2 = (a**2 - wr**2)/vr**2 if abs(vr) > 0.001 else 1.0
    k3 = 2*c*a
    u2 = - k2 * vr * np.sin(e3)/e3 * e2 - k3 * e3 if abs(e3) > 0.001 else 0.0
    w = wr - u2

    return v, w, u1, u2, e1, e2, e3

def simulator(Vr, Wr, Xr, Yr, Thetar, x0, y0, theta0, dt):
    odom_model = OdometryModel(x0, y0, np.deg2rad(theta0), noise=1e-3)
    x, y, theta = odom_model.get_state()
    logger = {"x": [], "y": [], "theta": [], "u1": [], "u2": [], "e1": [], "e2": [], "e3": []}
    for t in range(len(Vr)):
        v, w, u1, u2, e1, e2, e3 = nonlin_controller(x, y, theta, Xr[t], Yr[t], Thetar[t], Vr[t], Wr[t])
        #log x, y, theta, v, w, u1, u2, e1, e2, e3
        for name, val in zip(["x", "y", "theta", "u1", "u2", "e1", "e2", "e3"], [x, y, theta, u1, u2, e1, e2, e3]):
            logger[name].append(val)

        odom_model.update(v, w, dt)
        x, y, theta = odom_model.get_state()
    return logger

def make_plots(old_mrs, new_mrs, logger, theta_i, theta_f, k, block_size, s, dt, t, st, name=""):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    x, y = compute_center_path(old_mrs, new_mrs, theta_i, theta_f, k=k)
    ax[0, 0].plot(x(s),y(s), marker="^")
    ax[0, 0].set_aspect('equal')

    xa, ya = compute_robots_path(old_mrs, new_mrs, theta_i, theta_f, k=k) 
    mark = ["o", "s", "p"]
    for i, (rl, ra) in enumerate(old_mrs.agents):
        ax[0, 0].plot(xa(s, rl, ra), ya(s, rl, ra), color="grey")
        colors = pl.cm.inferno(s)
        for x, y, c in zip(xa(s, rl, ra), ya(s, rl, ra), colors):
            ax[0, 0].plot(x, y, marker=mark[i], color=c, alpha=.5)

    for mrs in [old_mrs, new_mrs]:
        agent = mrs.get_agents_positions()
        x1, y1 = agent[0]*block_size
        x2, y2 = agent[1]*block_size
        ax[0, 0].plot(x1, y1, marker="o", markersize=10, color="grey")
        ax[0, 0].plot(x2, y2, marker="o", markersize=10, color="grey")
        ax[0, 0].plot([x1, x2], [y1, y2], color="gray")
    ax[0, 0].set_title("reference robot's path")
    ax[0, 0].set_xlabel("[m]")
    ax[0, 0].set_ylabel("[m]")

    
    ax[0, 1].plot(t, np.rad2deg(logger["thetaref"]))
    ax[0, 1].set_title("reference heading angle")
    ax[0, 1].set_xlabel("time [s]")
    ax[0, 1].set_ylabel("angle [degree]")
    ax[0, 1].legend(["R1", "R2"])

    ax[0, 2].plot(t, logger["vref"])
    ax[0, 2].set_title("reference linear velocity")
    ax[0, 2].set_xlabel("time [s]")
    ax[0, 2].set_ylabel("velocity [m/s]")
    ax[0, 2].legend(["R1", "R2"])

    ax[0, 3].plot(t, logger["wref"])
    ax[0, 3].set_title("reference angular velocity")
    ax[0, 3].set_xlabel("time [s]")
    ax[0, 3].set_ylabel("angular velocity [rad/s]")
    ax[0, 3].legend(["R1", "R2"])

    ax[1, 0].plot(t, logger["e1"], label="e1")
    ax[1, 0].plot(t, logger["e2"], label="e2")
    ax[1, 0].set_title("position error")
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_ylabel("[m]")
    ax[1, 0].legend()

    ax[1, 1].plot(t, np.rad2deg(logger["e3"]), label="e3")
    ax[1, 1].set_title("heading error")
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylabel("[degree]")
    ax[1, 1].legend()

    ax[1, 2].plot(logger["x"], logger["y"], label="robot")
    ax[1, 2].plot(logger["xref"], logger["yref"], label="reference")
    ax[1, 2].set_title("robot's path")
    ax[1, 2].set_xlabel("[m]")
    ax[1, 2].set_ylabel("[m]")
    ax[1, 2].legend()

    ax[1, 3].plot(t, np.rad2deg(logger["theta"]), label="robot")
    ax[1, 3].plot(t, np.rad2deg(logger["thetaref"]), label="reference")
    ax[1, 3].set_title("robot's heading")
    ax[1, 3].set_xlabel("time [s]")
    ax[1, 3].set_ylabel("[degree]")
    ax[1, 3].legend()

    plt.tight_layout()
    #plt.savefig(f"pathplot_{name}.pdf")
    plt.show()

if __name__ == "__main__":
    block_size = 0.3
    k=0.2

    current_heading = 0
    cuurent_x = 3.15
    current_y = 1.95

    old_mrs = MRS(np.array([[1.5, 180], [1.5, 0]]), np.array([10.5, 8]), angle=-90, mass=2, inertia=0.08)
    for iter, directions in enumerate([np.array([[1, 0], [0, 1]]), 
                        np.array([[1, 0], [1, 0]]), 
                        np.array([[1, -1], [1, 0]]),
                        np.array([[1, -1], [1, 1]]),
                        np.array([[1, -1], [1, 1]]),
                        np.array([[1, 0], [1, 0]])]):
        new_mrs = old_mrs.move(directions)

        samples = 501
        s = np.linspace(0, 1, samples)
        T = 5
        dt = T/(samples-1)
        t = np.linspace(0, T, samples)
        st = t/T

        rl, ra = old_mrs.agents[1]
        robot_heading = np.rad2deg(np.arctan2(new_mrs.pos[1]-old_mrs.pos[1], new_mrs.pos[0]-old_mrs.pos[0]))
        print("HEADING", robot_heading)

        theta_i = np.deg2rad(old_mrs.angle + 90 + robot_heading)
        theta_f = np.deg2rad(new_mrs.angle + 90 + robot_heading)
        xref, yref = compute_robots_path(old_mrs, new_mrs, theta_i, theta_f, k=k) 
        xref, yref = xref(st, rl, ra), yref(st, rl, ra)
        theta_r1, v_r1, w_r1 = compute_reference_input(xref, yref, dt)

        samples_rot = 300
        theta_rot = []
        for i in range(samples_rot):
            theta_rot.append(current_heading*(1-(i/samples_rot)) + np.rad2deg(theta_r1[0])*i/samples_rot)
        xref, yref = np.concatenate([np.ones((samples_rot))*xref[0], xref]), np.concatenate([np.ones((samples_rot))*yref[0], yref]) 
        theta_r1, v_r1, w_r1 = np.concatenate([np.deg2rad(np.array(theta_rot)), theta_r1]), np.concatenate([np.zeros((samples_rot)), v_r1]), np.concatenate([np.ones((samples_rot))*np.deg2rad(np.rad2deg(theta_r1[0])-current_heading)/(samples_rot*dt), w_r1]) 

        samples = 801
        s = np.linspace(0, 1, samples)
        T = 8
        dt = T/(samples-1)
        t = np.linspace(0, T, samples)
        st = t/T
        logger = simulator(v_r1, w_r1, xref, yref, theta_r1, cuurent_x, current_y, current_heading, dt)
        logger["xref"], logger["yref"], logger["thetaref"], logger["vref"], logger["wref"] = xref, yref, theta_r1, v_r1, w_r1

        make_plots(old_mrs, new_mrs, logger, theta_i, theta_f, k, block_size, s, dt, t, st, name=str(iter))
        current_heading = np.rad2deg(logger["theta"][-1])
        cuurent_x = logger["x"][-1]
        current_y = logger["y"][-1]
        old_mrs = new_mrs