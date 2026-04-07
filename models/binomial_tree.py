import math

def price(s, k, r, t, sigma, n, option_type="call", exercise="european"): 
    option_type = option_type.lower()
    exercise = exercise.lower()

    if t == 0:
        if option_type == "call":
            return max(s - k, 0)
        elif option_type == "put":
            return max(k - s, 0)

    dt = t/n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1/u
    p = (math.exp(r * dt) - d) / (u - d)

    if option_type == "call":
        values = [max(s * u**i * d**(n - i) - k, 0) for i in range(n+1)]
    elif option_type == "put":
        values = [max(k - s * u**i * d**(n-i), 0) for i in range(n+1)]

    for step in range(n-1, -1, -1):
        for i in range(step+1):
            values[i] = math.exp(-r * dt) * (p * values[i+1] + (1-p) * values[i])      
            if exercise == "american":
                if option_type == "call":
                    values[i] = max(values[i], s * u**i * d**(step - i) - k)
                else:
                    values[i] = max(values[i], k - s * u**i * d**(step - i))

    return values[0]