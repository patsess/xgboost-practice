
import numpy as np
from skopt import BayesSearchCV
from skopt.utils import point_asdict

import skopt; assert (skopt.__version__ == '0.5.2')
# note: if skopt is updated then its _step() method could change (and this
# BayesSearchCV_ class may not be necessary anyway)


class BayesSearchCV_(BayesSearchCV):
    """Edited version of BayesSearchCV, in which an insurance against
    'n_points' being converted into a np.int64 (which can cause crashes)
    is added.
    """
    def _step(self, X, y, search_space, optimizer, groups=None, n_points=1):
        """Generate n_jobs parameters and evaluate them in parallel.
        """
        if isinstance(n_points, np.int64):
            n_points = int(n_points)
            # NOTE: THIS IS THE CODE ADDED TO BayesSearchCV (see class docstr)

        # get parameter values to evaluate
        params = optimizer.ask(n_points=n_points)
        params_dict = [point_asdict(search_space, p) for p in params]

        # HACK: self.cv_results_ is reset at every call to _fit, keep current
        all_cv_results = self.cv_results_

        # HACK: this adds compatibility with different versions of sklearn
        refit = self.refit
        self.refit = False
        self._fit(X, y, groups, params_dict)
        self.refit = refit

        # merge existing and new cv_results_
        for k in self.cv_results_:
            all_cv_results[k].extend(self.cv_results_[k])

        self.cv_results_ = all_cv_results
        self.best_index_ = np.argmax(self.cv_results_['mean_test_score'])

        # feed the point and objective back into optimizer
        local_results = self.cv_results_['mean_test_score'][-len(params):]

        # optimizer minimizes objective, hence provide negative score
        return optimizer.tell(params, [-score for score in local_results])
