from jinja2 import Environment, meta


def get_undeclared_variables(template: str) -> set[str]:
    """Get the variables from a Jinja2 template."""
    env = Environment(autoescape=True)
    parsed_content = env.parse(template)
    variables = meta.find_undeclared_variables(parsed_content)

    return variables
