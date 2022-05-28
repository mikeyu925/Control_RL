import numpy as np
import servo_system_env


t = np.random.normal(0, 0.001, size=2)
# [0.09631596 0.12487053]
# [0.01697105 0.00944596]
print(t)
print(t.shape)

print('*******************')
env = servo_system_env.ServoSystemEnv()
print(env.state_dim)