module MXNet
  module Random
    module_function

    # Draw random samples from a uniform distribution
    #
    # Samples are uniformly distributed over the half-open interval *[low, high)*
    # (includes *low*, but excludes *high*).
    def uniform(low=0, high=1, shape: nil, dtype: nil, ctx: nil, out: nil, **kwargs)
      _random_helper(:_random_uniform, :_sample_uniform, {low: low, high: high}, shape, dtype, ctx, out, kwargs)
    end

    # Draw random samples from a normal (Gaussian) distribution.
    #
    # Samples are distributed according to a normal distribution parametrized
    # by *loc* (mean) and *scale* (standard deviation).
    def normal(loc=0, scale=1, shape: nil, dtype: nil, ctx: nil, out: nil, **kwargs)
      _random_helper(:_random_normal, :_sample_normal, {loc: loc, scale: scale}, shape, dtype, ctx, out, kwargs)
    end

    def poisson(lam=1, shape: nil, dtype: nil, ctx: nil, out: nil, **kwargs)
      _random_helper(:_random_poisson, :_sample_poisson, {lam: lam}, shape, dtype, ctx, out, kwargs)
    end

    def exponential(scale=1, shape: nil, dtype: nil, ctx: nil, out: nil, **kwargs) 
      _random_helper(:_random_exponential, :_sample_exponential, {scale: scale}, shape, dtype, ctx, out, kwargs)
    end

    def gamma(alpha=1, beta=1, shape: nil, dtype: nil, ctx: nil, out: nil, **kwargs) 
      _random_helper(:_random_gamma, :_sample_gamma, {alpha: alpha, beta: beta}, shape, dtype, ctx, out, kwargs)
    end

    def negative_binomial(k=1, p=1, shape: nil, dtype: nil, ctx: nil, out: nil, **kwargs) 
      _random_helper(:_random_negative_binomial, :_sample_negative_binomial,
                     {k: k, p: p}, shape, dtype, ctx, out, kwargs)
    end

    def generalized_negative_binomial(mu=1, alpha=1, shape: nil, dtype: nil, ctx: nil, out: nil, **kwargs) 
      _random_helper(:_random_generalized_negative_binomial,
                     :_sample_generalized_negative_binomial,
                     {mu: mu, alpha: alpha}, shape, dtype, ctx, out, kwargs)
    end

    # Concurrent sampling from multiple multinomial distributions.
    #
    # NOTE: The input distribution must be normalized,
    #       i.e. `data` must sum to 1 along its last dimension.
    def multinomial(data, shape: nil, dtype: nil, get_prob: false, out: nil, **kwargs) 
      NDArray::Internal._sample_multinomial(data, shape, get_prob, out: out, **kwargs)
    end

    def _random_helper(random, sampler, params, shape, dtype, ctx, out, kwargs)
      first_value, *rest_values = params.values
      if first_value.is_a? NDArray
        rest_values.each do |i|
          unless i.is_a? NDArray
            raise ArgumentError, "Distributed parameters must all have the same type, but got " +
              "both #{first_value.class} and #{i.class}"
          end
        end
        NDArray::Internal.send(sampler, first_value, *rest_values, shape: shape, dtype: dtype, out: out, **kwargs)
      elsif first_value.is_a? Numeric
        ctx = Context.current if ctx.nil?
        shape = 1 if shape.nil? && out.nil?
        rest_values.each do |i|
          unless i.is_a? Numeric
            raise ArgumentError, "Distributed parameters must all have the same type, but got " +
              "both #{first_value.class} and #{i.class}"
          end
        end
        NDArray::Internal.send(random, **params, shape: shape, dtype: dtype, out: out, **kwargs)
      else
        raise ArgumentError, "Distribution parameters must be either NDArray or Numeric, " +
          "but got #{params[0].class}."
      end
    end
  end
end
