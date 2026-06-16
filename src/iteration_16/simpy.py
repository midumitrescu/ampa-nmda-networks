import unittest

from brian2 import nS, ms, mV, nF
from sympy import solve

from iteration_16.model import calibrated_configuration, Chapter1Results
import sympy as sp


class SimpyTest(unittest.TestCase):

    def test_system_elimination_1(self):
        cfg = calibrated_configuration

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
            we: float(cfg.w_ampa / nS),
            wi: float(cfg.w_gaba / nS),
            taue: float(cfg.tau_ampa / ms),
            taui: float(cfg.tau_gaba / ms),
            C: float(cfg.membrane_capacitance / nF)
        })
        E_target = float(Chapter1Results.mu_v / mV)

        eq1 = E0_expr - E_target

        eq2 = sigma2.subs({
            gL: float(cfg.g_L / nS),
            EL: float(cfg.e_L / mV),
            Ee: float(cfg.e_ampa / mV),
            Ei: float(cfg.e_gaba / mV),
            we: float(cfg.w_ampa / nS),
            wi: float(cfg.w_gaba / nS),
            taue: float(cfg.tau_ampa / ms),
            taui: float(cfg.tau_gaba / ms),
            C: float(cfg.membrane_capacitance / nF)
        }) - (Chapter1Results.sigma_v / mV) ** 2

        sol = sp.nsolve(
            (eq1, eq2),
            (x, gamma),
            (100.0, 4.0)  # initial guesses
        )

        print(sol)

    def test_system_elimination_2(self):
        from sympy.abc import x, y

        # x = gr, y = gamma
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

        equation_sigma_sq = sigma_sq - sigma_target

        y_expr = sp.solve(equation_e_0, y)[0]
        print(sp.factor(y_expr))
        sp.pretty(y_expr)

        eq_x = sp.simplify(
            equation_sigma_sq.subs(y, y_expr)
        )

        print(eq_x)

        # Cancel common factors first
        eq_x = sp.cancel(eq_x)

        num, den = sp.fraction(eq_x)

        print("Numerator degree:", sp.degree(num, x))
        print("Denominator degree:", sp.degree(den, x))
        print()

        sol = sp.solve(num, x)
        print(sol)
        with open("solution.txt", "w") as f:
            f.write(sp.srepr(sol))
        print("=========================================================================")
        print("=========================================================================")
        x_expr = sp.expand(eq_x)
        print("Expansion: ", x_expr)
        P = sp.Poly(x_expr, x)

        print(P)

        eq_x = sp.cancel(eq_x)
        num, den = sp.fraction(eq_x)
        solve(num, x)

        print(solve([equation_e_0, equation_sigma_sq], (x, y)))

    def test_solution_read(self):
        with open("solution.txt") as f:
            solutions = sp.sympify(f.read())
            cfg = calibrated_configuration

            values = {
                'gL': float(cfg.g_L / nS),
                "EL": float(cfg.e_L / mV),
                "Ee": float(cfg.e_ampa / mV),
                "Ei": float(cfg.e_gaba / mV),
                "we": float(cfg.w_ampa / nS),
                "wi": float(cfg.w_gaba / nS),
                "taue": float(cfg.tau_ampa / ms),
                "taui": float(cfg.tau_gaba / ms),
                "C": float(cfg.membrane_capacitance / nF),
                'sigma_target': float(Chapter1Results.sigma_v / mV),
                'E_target': float(Chapter1Results.mu_v / mV)
            }

            for index, solution in enumerate(solutions):
                # substitute by symbol name
                print("===============================")
                print("Solution # ", index)
                x_val = solution.subs(values)

                #print("symbolic:", x_val)

                # numerical evaluation
                x_val = x_val.evalf()
                if abs(sp.im(x_val)) < 1e-10:
                    x_val = sp.re(x_val)

                print("numeric:", x_val)
                print("===============================")

                y_expr = sp.solve(equation_e_0, y)[0]

if __name__ == '__main__':
    unittest.main()
