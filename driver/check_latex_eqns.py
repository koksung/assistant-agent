from app.tools.local.equation_renderer import parse_latex_equation


def test_equations():
    # print("=== Testing Manual Equation Creation ===")
    # create_ddpm_equations()

    print("\n=== Testing LaTeX Parsing ===")
    local_test_equations = [
        r"\alpha_t = 1 - \beta_t",
        r"x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon",
        r"\frac{\partial L}{\partial x_t}",
    ]

    for eq in local_test_equations:
        print(f"\nInput LaTeX: {eq}")
        print("Parsed result:")
        print(parse_latex_equation(eq))


if __name__ == "__main__":
    # Test basic functionality
    test_equations()
