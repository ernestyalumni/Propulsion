from math import pow

class ComputePIStepSize:

    def __init__(
            self,
            alpha,
            beta,
            safety_factor = 0.9,
            min_scale = 0.2,
            max_scale = 5.0):
        self.alpha = alpha
        self.beta = beta
        self.safety_factor = safety_factor
        self.min_scale = min_scale
        self.max_scale = max_scale

    def compute_new_step_size(self, error, h, was_rejected):
        if error <= 1:

            scale = self.max_scale if error == 0 else (
                self.safety_factor *
                    pow(error, -self.alpha) * pow(error, self.beta))

            # Ensure min_scale <= h_new / h <= max_scale

            scale = min(max(scale, self.min_scale), self.max_scale)

            return h * min(scale, 1) if was_rejected else h * scale

        scale = max(self.safety_factor * pow(error, -alpha), self.min_scale)
        return h * scale