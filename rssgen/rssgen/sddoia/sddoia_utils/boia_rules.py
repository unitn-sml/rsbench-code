# Module which contains the rules for SDDOIA
# SDDOIAK is the default logic
# oodSDDoiaK is the ood logic, in case specified

from sympy import symbols, Not, Or, Eq, Implies, And, Piecewise, sympify


def sddoiaK(
    red_light,
    green_light,
    car,
    person,
    rider,
    other_obstacle,
    follow,
    stop_sign,
    left_lane,
    left_green_light,
    left_follow,
    no_left_lane,
    left_obstacle,
    left_solid_line,
    right_lane,
    right_green_light,
    right_follow,
    no_right_lane,
    right_obstacle,
    right_solid_line,
):
    # Define the formulas
    red_light = And(Implies(green_light, Not(red_light)), red_light)
    obstacle = Or(car, person, rider, other_obstacle)
    road_clear = Not(obstacle)
    move_forward_cond = Or(green_light, follow, road_clear)
    stop = Or(red_light, stop_sign, obstacle)
    move_forward = And(Implies(stop, Not(move_forward_cond)), move_forward_cond)

    # can turn left and can turn right as rules
    can_turn_left = Or(left_lane, left_green_light, left_follow)
    cannot_turn_left = Or(no_left_lane, left_obstacle, left_solid_line)
    turn_left = And(can_turn_left, Not(cannot_turn_left))

    # can turn right and cannot turn right as rules
    can_turn_right = Or(right_lane, right_green_light, right_follow)
    cannot_turn_right = Or(no_right_lane, right_obstacle, right_solid_line)
    turn_right = And(can_turn_right, Not(cannot_turn_right))

    # return multilabel classification element
    return (
        [stop, move_forward, turn_left, turn_right],
        road_clear,
        move_forward_cond,
        stop,
    )


# SYMBOLS
red_light = symbols("red_light")
green_light = symbols("green_light")
car = symbols("car")
person = symbols("person")
rider = symbols("rider")
other_obstacle = symbols("other_obstacle")
follow = symbols("follow")
stop_sign = symbols("stop_sign")
left_lane = symbols("left_lane")
left_green_light = symbols("left_green_light")
left_follow = symbols("left_follow")
no_left_lane = symbols("no_left_lane")
left_obstacle = symbols("left_obstacle")
left_solid_line = symbols("left_solid_line")
right_lane = symbols("right_lane")
right_green_light = symbols("right_green_light")
right_follow = symbols("right_follow")
no_right_lane = symbols("no_right_lane")
right_obstacle = symbols("right_obstacle")
right_solid_line = symbols("right_solid_line")


def apply_sddoiaK(values):
    (stop, move_forward, turn_left, turn_right), clear, mf, stop = sddoiaK(
        red_light,
        green_light,
        car,
        person,
        rider,
        other_obstacle,
        follow,
        stop_sign,
        left_lane,
        left_green_light,
        left_follow,
        no_left_lane,
        left_obstacle,
        left_solid_line,
        right_lane,
        right_green_light,
        right_follow,
        no_right_lane,
        right_obstacle,
        right_solid_line,
    )

    # Substitute the symbols with their actual values and convert the results to integers
    result = [
        int(bool(stop.subs(values))),
        int(bool(move_forward.subs(values))),
        int(bool(turn_left.subs(values))),
        int(bool(turn_right.subs(values))),
    ]

    clear = bool(clear.subs(values))

    return result, clear


def oodSDDoiaK(
    red_light,
    green_light,
    car,
    person,
    rider,
    other_obstacle,
    follow,
    stop_sign,
    left_lane,
    left_green_light,
    left_follow,
    no_left_lane,
    left_obstacle,
    left_solid_line,
    right_lane,
    right_green_light,
    right_follow,
    no_right_lane,
    right_obstacle,
    right_solid_line,
):
    # AMBULANCE RULES

    # Define the formulas
    red_light = And(Implies(green_light, Not(red_light)), red_light)
    obstacle = Or(car, person, rider, other_obstacle)

    # the ambulance stops only when there is something in the middle of the street, otherwise
    # it proceeds
    stop = Or(car, person, rider, other_obstacle)
    # In all other cases the car should proceed
    move_forward = Not(stop)

    # can turn left and can turn right as rules
    can_turn_left = left_lane  # the follow and green light can be ignored
    cannot_turn_left = Or(no_left_lane, left_obstacle)  # the solid line can be ignored
    turn_left = And(can_turn_left, Not(cannot_turn_left))

    # can turn right and cannot turn right as rules
    can_turn_right = right_lane  # the follow and green light can be ignored
    cannot_turn_right = Or(
        no_right_lane, right_obstacle
    )  # the solid line can be ignored
    turn_right = And(can_turn_right, Not(cannot_turn_right))

    # return multilabel classification element
    return [stop, move_forward, turn_left, turn_right]


def ood_knowledge(values):
    (stop, move_forward, turn_left, turn_right) = oodSDDoiaK(
        red_light,
        green_light,
        car,
        person,
        rider,
        other_obstacle,
        follow,
        stop_sign,
        left_lane,
        left_green_light,
        left_follow,
        no_left_lane,
        left_obstacle,
        left_solid_line,
        right_lane,
        right_green_light,
        right_follow,
        no_right_lane,
        right_obstacle,
        right_solid_line,
    )

    # Substitute the symbols with their actual values and convert the results to integers
    result = [
        int(bool(stop.subs(values))),
        int(bool(move_forward.subs(values))),
        int(bool(turn_left.subs(values))),
        int(bool(turn_right.subs(values))),
    ]

    return result
