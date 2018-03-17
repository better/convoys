import bisect
import lifelines
import numpy

class KaplanMeier:
    def fit(self, X, B, T):
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(T, event_observed=B)
        self.ts = kmf.survival_function_.index.values
        self.ps = 1.0 - kmf.survival_function_['KM_estimate'].values
        self.ps_hi = 1.0 - kmf.confidence_interval_['KM_estimate_lower_0.95'].values
        self.ps_lo = 1.0 - kmf.confidence_interval_['KM_estimate_upper_0.95'].values

    def predict(self, x, ts, ci=None):
        js = [bisect.bisect_left(self.ts, t) for t in ts]
        def array_lookup(a):
            return numpy.array([a[j] for j in js if j < len(self.ts)])
        if ci is not None:
            return (array_lookup(self.ts), array_lookup(self.ps), array_lookup(self.ps_lo), array_lookup(self.ps_hi))
        else:
            return (array_lookup(self.ts), array_lookup(self.ps))

    def predict_final(self, x, ci=None):
        if ci is not None:
            return (self.ps[-1], self.ps_lo[-1], self.ps_hi[-1])
        else:
            return self.ps[-1]

    def predict_time(self, x, ci=None):
        # TODO: should not use median here, but mean is no good
        def median(ps):
            i = bisect.bisect_left(ps, 0.5)
            return self.ts[min(i, len(ps)-1)]
        if ci is not None:
            return median(self.ps), median(self.ps_lo), median(self.ps_hi)
        else:
            return median(self.ps)
