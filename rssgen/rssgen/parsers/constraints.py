"""List of constraints for the parsers"""


def greater_than_zero(value):
    return value > 0


def greater_than_one(value):
    return value > 1


def between_zero_one(value):
    return value >= 0 and value <= 1


def between_zero_nine(value):
    return value >= 0 and value <= 9


def len_not_zero(a_list):
    return len(a_list) > 0


def list_between_zero_one(a_list):
    for a in a_list:
        if not between_zero_one(a):
            return False
    return True


def list_between_zero_nine(a_list):
    for a in a_list:
        if not between_zero_nine(a):
            return False
    return True
