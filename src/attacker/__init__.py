from attacker.AUTOConjugate import AUTOConjugate

import utils

logger = utils.setup_logger(__name__)


attacker_list = [
    "AUTOConjugate"
]


def Attackers(attacker_name, *args, **kwargs):
    """
    Parameters
    ----------
    attacker_name : src
        attacker name
    Return
    ------
    Attacker

    """
    if attacker_name == "AUTOConjugate":
        return AUTOConjugate(*args, **kwargs)
    else:
        message = (
            f"attacker must be select from {attacker_list}",
            f", but got {attacker_name}",
        )
        logger.error(message)
        raise NotImplementedError
