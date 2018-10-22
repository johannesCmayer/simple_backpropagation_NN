
def derive():
    pass

def chain_rule(outer, inner):
    return derive(outer) * derive(inner)

def product_rule(f_1, f_2):
    return derive(f_1) * f_2 + f_1 * derive(f_2)

def add_rule(f_1, f_2):
    return derive(f_1) + derive(f_2)