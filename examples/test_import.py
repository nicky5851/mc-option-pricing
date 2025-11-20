print('trying to import mcopt....')

import mcopt

print('Success, mcopt was imported!!!')

from mcopt.models.gbm import GBM

gbm = GBM(mu=0.05, sigma=0.2)
print("GBM created:", gbm.mu, gbm.sigma)
