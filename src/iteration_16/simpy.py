import unittest

import sympy as sp
from brian2 import nS, mV, nF, second
from sympy import solve, Reals
from sympy.physics.control.control_plots import plt
from sympy.physics.quantum.identitysearch import np

from Plotting import prepare_bigger_fonts
from iteration_16.model import config_with_weak_synapses, chapter1Results, config_with_medium_synapses

def bind_config_to_sympy_values(cfg):
    values = {
        'gL': float(cfg.g_L / nS),
        "EL": float(cfg.e_L / mV),
        "Ee": float(cfg.e_ampa / mV),
        "Ei": float(cfg.e_gaba / mV),
        "we": float(cfg.w_ampa / nS),
        "wi": float(cfg.w_gaba / nS),
        "taue": float(cfg.tau_ampa / second),
        "taui": float(cfg.tau_gaba / second),
        "C": float(cfg.membrane_capacitance / nF),
        'sigma_target': float(chapter1Results.sigma_v / mV),
        'E_target': float(chapter1Results.mu_v / mV)
    }
    return values

class SimpyTest(unittest.TestCase):

    def test_system_elimination_1(self):
        cfg = config_with_weak_synapses

        x = sp.symbols('x')

        gL, Ee, Ei, EL = sp.symbols('gL Ee Ei EL')
        gamma = sp.symbols('gamma')
        we, wi = sp.symbols('we wi')
        taue, taui = sp.symbols('taue taui')
        C = sp.symbols('C')

        g0 = gL + x * (1 + gamma)

        E0 = (gL * EL + x * Ee + gamma * x * Ei) / g0

        sigma2 = (
                we / 2 * x / g0 ** 2 * (Ee - E0) ** 2 *
                taue / (taue + C / g0)
                +
                wi / 2 * gamma * x / g0 ** 2 * (Ei - E0) ** 2 *
                taui / (taui + C / g0)
        )
        sigma2_simplified = sp.together(sp.simplify(sigma2))
        num, den = sp.fraction(sigma2_simplified)

        print(sp.factor(num))
        print(sp.factor(den))

        sigma_target = sp.symbols('sigma_target')

        E0_expr = E0.subs({
            gL: float(cfg.g_L / nS),
            EL: float(cfg.e_L / mV),
            Ee: float(cfg.e_ampa / mV),
            Ei: float(cfg.e_gaba / mV),
        })
        E_target = float(chapter1Results.mu_v / mV)

        eq1 = E0_expr - E_target

        eq2 = sigma2.subs({
            gL: float(cfg.g_L / nS),
            EL: float(cfg.e_L / mV),
            Ee: float(cfg.e_ampa / mV),
            Ei: float(cfg.e_gaba / mV),
            we: float(cfg.w_ampa / nS),
            wi: float(cfg.w_gaba / nS),
            taue: float(cfg.tau_ampa / second),
            taui: float(cfg.tau_gaba / second),
            C: float(cfg.membrane_capacitance / nF)
        }) - (chapter1Results.sigma_v / mV) ** 2

        sol = sp.nsolve(
            (eq1, eq2),
            (x, gamma),
            (100.0, 4.0)  # initial guesses
        )

        print(sol)

    def test_system_elimination_2(self, produce_latex = False):
        from sympy.abc import x, y

        # x = gr, y = gamma
        gL, Ee, Ei, EL, E_target = sp.symbols('gL Ee Ei EL E_target')
        we, wi = sp.symbols('we wi')
        taue, taui = sp.symbols('taue taui')
        C = sp.symbols('C')
        sigma_target = sp.symbols('sigma_target')

        g0 = gL + x * (1 + y)

        cfg = config_with_weak_synapses
        subs ={
            gL: float(cfg.g_L / nS),
            EL: float(cfg.e_L / mV),
            Ee: float(cfg.e_ampa / mV),
            Ei: float(cfg.e_gaba / mV),
            we: float(cfg.w_ampa / nS),
            wi: float(cfg.w_gaba / nS),
            taue: float(cfg.tau_ampa / second),
            taui: float(cfg.tau_gaba / second),
            C: float(cfg.membrane_capacitance / nF),
            E_target: float(chapter1Results.mu_v / mV),
            sigma_target: float(chapter1Results.sigma_v / mV),
        }

        E0 = (gL * EL + x * Ee + y * x * Ei) / g0
        equation_e_0 = E0 - E_target
        sigma_sq = (
                we / 2 * x / g0 ** 2 * (Ee - E0) ** 2 *
                taue / (taue + C / g0)
                +
                wi / 2 * y * x / g0 ** 2 * (Ei - E0) ** 2 *
                taui / (taui + C / g0)
        )

        equation_sigma_sq = sigma_sq - sigma_target**2

        y_expr = sp.solve(equation_e_0, y)[0]
        print(sp.pretty(y_expr))


        eq_x = sp.simplify(
            equation_sigma_sq.subs(y, y_expr)
        )

        eq_x = sp.cancel(eq_x)

        expr = sp.together(eq_x)
        num, _ = expr.as_numer_denom()

        num = sp.collect(num, x)
        poly = sp.Poly.from_expr(num, x)

        if produce_latex:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(sp.latex(num, order='grlex'))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Roots ", sp.roots(poly.as_expr(), x))
        print("Roots ",sp.nroots(poly.subs(subs).as_expr()))

        with open("polynomial.txt", "w") as f:
            f.write(sp.srepr(poly.as_expr()))

        sol = sp.solve(num, x, domain=Reals)
        print(sol)

        with open("solution.txt", "w") as f:
            f.write(sp.srepr(sol))

        print("=========================================================================")
        print("=========================================================================")

        eq_x = sp.cancel(eq_x)
        num, den = sp.fraction(eq_x)
        solve(num, x)

        print(solve([equation_e_0, equation_sigma_sq], (x, y)))

    def test_solution_read(self):
        from sympy.abc import x, y

        gL, Ee, Ei, EL, E_target = sp.symbols('gL Ee Ei EL E_target')
        g0 = gL + x * (1 + y)

        E0 = (gL * EL + x * Ee + y * x * Ei) / g0

        we, taue, wi, taui, C = sp.symbols('we taue wi taui C')

        sigma_sq = (
                we / 2 * x / g0 ** 2 * (Ee - E0) ** 2 *
                taue / (taue + C / g0)
                +
                wi / 2 * y * x / g0 ** 2 * (Ei - E0) ** 2 *
                taui / (taui + C / g0)
        )
        equation_e_0 = E0 - E_target
        y_expr = sp.solve(equation_e_0, y)[0]

        with open("solution.txt") as f:
            solutions = sp.sympify(f.read())
            cfg = config_with_weak_synapses

            values = self.bind_config_to_sympy_values(cfg)

            for index, solution in enumerate(solutions):
                x_val = solution.subs(values).evalf()
                if abs(sp.im(x_val)) < 1e-10:
                    x_val = sp.re(x_val)

                y_val = y_expr.subs(values).subs(x, x_val)

                self.assertAlmostEqual(-47.6159564500000, float(E0.subs(values).subs({"x": x_val, "y": y_val})), places=7)
                self.assertAlmostEqual(3.60999999999985, float(sigma_sq.subs(values).subs({"x": x_val, "y": y_val})), places=7)

    def test_polynomial_plot(self):
        from sympy.abc import x, y
        gL, Ee, Ei, EL, E_target = sp.symbols('gL Ee Ei EL E_target')
        we, wi = sp.symbols('we wi')
        taue, taui = sp.symbols('taue taui')
        C = sp.symbols('C')
        sigma_target = sp.symbols('sigma_target')

        g0 = gL + x * (1 + y)

        E0 = (gL * EL + x * Ee + y * x * Ei) / g0
        equation_e_0 = E0 - E_target
        sigma_sq = (
                we / 2 * x / g0 ** 2 * (Ee - E0) ** 2 *
                taue / (taue + C / g0)
                +
                wi / 2 * y * x / g0 ** 2 * (Ei - E0) ** 2 *
                taui / (taui + C / g0)
        )

        equation_sigma_sq = sigma_sq - sigma_target ** 2

        y_expr = sp.solve(equation_e_0, y)[0]
        eq_x = sp.simplify(
            equation_sigma_sq.subs(y, y_expr)
        )
        # Cancel common factors first
        eq_x = sp.cancel(eq_x)

        num, den = sp.fraction(eq_x)

        print("Numerator degree:", sp.degree(num, x))
        print("Denominator degree:", sp.degree(den, x))
        print()

        values_weak_synapses = bind_config_to_sympy_values(config_with_weak_synapses)
        values_moderate_synapses = bind_config_to_sympy_values(config_with_medium_synapses)        polynomial = sp.factor(num)
        P_weak = polynomial.subs(values_weak_synapses)
        f_weak = sp.lambdify(x, P_weak, "numpy")

        P_moderate = polynomial.subs(values_moderate_synapses)
        f_moderate = sp.lambdify(x, P_moderate, "numpy")

        for x_max in [300, 2000]:
            x_minus = np.linspace(-200, 0, 1000)
            x_plus = np.linspace(0, x_max, 1000)

            plt.figure(figsize=(10, 8))
            plt.plot(x_minus, f_weak(x_minus), color="red", linewidth=2, linestyle="-.", label=r"negative conductances, weak synapses")
            plt.plot(x_plus, f_weak(x_plus), linewidth=2, label=fr"positive conductances, weak synapses, γ = {y}")
            plt.plot(x_minus, f_moderate(x_minus), color="red", linewidth=2, linestyle="-.", label=r"negative conductances, moderate synapses")
            plt.plot(x_plus, f_moderate(x_plus), linewidth=2, label=r"positive conductances, moderate synapses")
            plt.axhline(y=0, linestyle='--', color='k')

            plt.xlabel("$x \equiv gr$ (nS)")
            plt.ylabel("$P(gr)$")

            plt.title("Solution for P(gr) = 0 after elimination of $\gamma$ \n"
                        r""" $g_0 = g_L + g \cdot r \cdot (1 + \gamma)$ """ "\n"
                        r"""$ \bar E_0 = \frac{g_L \cdot E_L + g \cdot r \cdot \left( E_e + \gamma \cdot  E_i \right)}{g_L  + g \cdot r \cdot (1 + \gamma)}$""""\n"
                        r"""$\bar\sigma_0^2 = \frac{w_e}{2} \cdot \frac{g \cdot r}{g_0} \cdot \bar E_e^2 \cdot \frac{\tau_e}{\tau_e g_0+C} + \frac{w_i}{2} \frac{\gamma \cdot g \cdot r}{g_0} \bar E_i ^2 \frac{ \tau_i} {\tau_i g_0+C}$""""\n")

            plt.legend()
            prepare_bigger_fonts(zoom=2)
            plt.tight_layout()
            plt.show()



    def test_unit_of_variable_x(self):

        one_solution


if __name__ == '__main__':
    unittest.main()
