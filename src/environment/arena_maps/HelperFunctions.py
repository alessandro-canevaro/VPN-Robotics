def get_ros_package_path(package_name: str) -> str:
    try:
        import rospkg

        rospack = rospkg.RosPack()
        return rospack.get_path(package_name)
    except:
        return ""


def get_nth_decimal_part(x: float, n: int) -> int:
    """
    Get the n'th decimal part of a decimal number.
    Example:
        get_nth_decimal_part(1.234, 2) == 3
    """
    x *= 10 ** n  # push relevant part directly in front of decimal point
    x %= 10  # remove parts left of the relevant part
    return int(x)  # remove decimal places


def round_to_closest_20th(x: float) -> float:
    """
    Round to X.X0 or X.X5.
    Example:
        round_one_and_half_decimal_places(1.234) == 1.25
    """
    return round(x * 20) / 20

def rad_to_deg(angle: float) -> float:
    import math
    angle = normalize_angle_rad(angle)
    angle = 360.0 * angle / (2.0 * math.pi)
    return angle

def deg_to_rad(angle: float) -> float:
    import math
    angle = normalize_angle_deg(angle)
    angle = 2 * math.pi * angle / 360.0
    return angle

def normalize_angle_deg(angle: float) -> float:
    import math

    # make sure angle is positive
    while angle < 0:
        angle += 360

    # make sure angle is between 0 and 360
    angle = math.fmod(angle, 360.0)
    return angle

def normalize_angle_rad(angle: float) -> float:
    import math

    # make sure angle is positive
    while angle < 0:
        angle += 2 * math.pi
    # make sure angle is between 0 and 2 * pi
    angle = math.fmod(angle, 2 * math.pi)
    return angle

def normalize_angle(angle: float, rad: bool = True) -> float:
    if rad:
        return normalize_angle_rad(angle)
    else:
        return normalize_angle_deg(angle)

def get_current_user_path(path_in: str) -> str:
    """
    Convert a path from another user to the current user, for example:
    "/home/alice/catkin_ws" -> "/home/bob/catkin_ws"
    """
    if path_in == "":
        return ""
    from pathlib import Path

    path = Path(path_in)
    new_path = Path.home().joinpath(*path.parts[3:])
    return str(new_path)

def remove_file_ending(file_name: str) -> str:
    """
    Remove everything after the first "." in a string.
    """
    file_ending_index = file_name.find(".")
    if file_ending_index != -1:
        return file_name[:file_ending_index]
    return file_name