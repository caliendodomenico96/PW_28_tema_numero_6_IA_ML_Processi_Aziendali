# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


""" Maths functions """


import math


def lognorm(rng, media, cv):
    """Distribuzione positiva con media e coefficiente di variazione indicati."""
    sigma2 = math.log1p(cv * cv)
    return rng.lognormvariate(math.log(media) - sigma2 / 2, math.sqrt(sigma2))