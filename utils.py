def rk4(f, t, x, u, dt):
    k1 = f(t, x, u)
    k2 = f(t + dt / 2, x + dt * k1 / 2, u)
    k3 = f(t + dt / 2, x + dt * k2 / 2, u)
    k4 = f(t + dt, x + dt * k3, u)

    return x + (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
