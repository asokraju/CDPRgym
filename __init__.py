import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="CDPR-v0", entry_point="CDPRgym.envs.cdpr:CDPRenv")

# gym.envs.register(
#  id='BicycleKin-v0',
#  entry_point='gym_bicycle.envs:BicycleKin'
# ) 
