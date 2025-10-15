import matplotlib.pyplot as plt
import matplotlib

print(matplotlib.get_backend())

plt.plot([0, 1], [0, 1])
plt.title(r"$E = mc^2$")
plt.xlabel(r"$\alpha + \beta = \gamma$")
plt.ylabel(r"$\int_a^b f(x)\,dx$")
plt.show()
