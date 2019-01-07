require 'mxnet/gluon/block'
require 'mxnet/gluon/nn'

module MXNet::Gluon::NN
  ##
  # Stacks blocks sequentially.
  #
  #     net = MXNet::Gluon::NN.Sequential
  #     net.with_name_scope do
  #       net.add(MXNet::Gluon::NN.Dense(10, activation: :relu))
  #       net.add(MXNet::Gluon::NN.Dense(20))
  #     end
  #
  class Sequential < MXNet::Gluon::Block
    ##
    # Creates a new instance.
    #
    def initialize(**kwargs)
      super(**kwargs)
    end

    ##
    # Adds blocks on top of the stack.
    #
    def add(*blocks)
      blocks.each { |block| register_child(block) }
    end

    def <<(block)
      add(block)
      self
    end

    ##
    # Runs a forward pass on all child blocks.
    #
    def forward(data)
      @children.each_value.inject(data) { |data, child| child.(data) }
    end
  end

  ##
  # Just your regular densely-connected neural network layer.
  #
  # Implements the operation:
  #
  #     output = activation(dot(input, weight) + bias)
  #
  # where +activation+ is the element-wise activation function passed
  # as the +activation+ argument, +weight+ is a weights matrix created
  # by the layer, and +bias+ is a bias vector created by the layer
  # (if argument +use_bias+ is +true+).
  #
  # Note: +input+ must be a tensor with rank two. Use
  # MXNet::NDArray#flatten to convert it to rank two if necessary.
  #
  class Dense < MXNet::Gluon::HybridBlock
    ##
    # Creates a new instance.
    #
    # ====Parameters
    #
    # +units_::      (integer)
    #                Dimensionality of the output space.
    # +use_bias+::   (boolean, default +true+)
    #                Whether the layer uses a bias vector.
    # +activation+:: (string, optional)
    #                Activation function to use. If nothing is
    #                specified, no activation is applied (it acts like
    #                "linear" activation: <tt>a(x) = x</tt>).
    # +in_units+::   (integer, optional)
    #                Size of the input data. If not specified,
    #                initialization will be deferred to the first time
    #                #forward is called and +in_units+ will be
    #                inferred from the shape of input data.
    #
    def initialize(units, use_bias: true, activation: nil, in_units: 0, **kwargs)
      super(**kwargs)
      with_name_scope do
        @units = units
        @use_bias = use_bias
        @in_units = in_units
        self[:weight] = params.get(
          'weight',
          shape: [units, in_units],
          allow_deferred_init: true
        )
        if @use_bias
          self[:bias] = params.get(
            'bias',
            shape: [units],
            allow_deferred_init: true
          )
        else
          self[:bias] = nil
        end
      end
      if activation
        self[:act] = MXNet::Gluon::NN::Activation(
          activation,
          prefix: "#{activation}_"
        )
      else
        self[:act] = nil
      end
    end

    def hybrid_forward(clazz, data, weight=nil, bias = nil, **kwargs)
      if (weight ||= kwargs[:weight]).nil?
        raise ArgumentError, "weight is not given"
      end
      bias ||= kwargs[:bias]
      out = clazz.FullyConnected(
        data,
        weight,
        bias,
        no_bias: bias.nil?,
        num_hidden: @units
      )
      if self[:act]
        out = self[:act].(out)
      end
      out
    end
  end

  def self.Dense(*args)
    Dense.new(*args)
  end

  module Internal
    ##
    # Base class for convolution layers.
    #
    # This layer creates a convolution kernel that is convolved with
    # the input to produce a tensor of outputs.
    #
    class Conv < MXNet::Gluon::HybridBlock
      ##
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +channels+::    (integer)
      #                 The dimensionality of the output space (the
      #                 number of output channels in the convolution).
      # +kernel_size+:: (array of N integers)
      #                 Specifies the dimensions of the convolution
      #                 window.
      # +strides+::     (integer or array of N integers)
      #                 Specifies the strides of the convolution.
      # +padding+::     (integer or array of N integers)
      #                 If padding is non-zero, then the input is
      #                 implicitly zero-padded on both sides for
      #                 +padding+ number of points.
      # +dilation+::    (integer or array of N integers)
      #                 Specifies the dilation rate to use for dilated
      #                 convolution.
      # +layout+::      (string)
      #                 Dimension ordering of data and weight. Can be
      #                 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW', 'NDHWC',
      #                 etc. 'N', 'C', 'H', 'W', 'D' stands for batch,
      #                 channel, height, width and depth dimensions
      #                 respectively. Convolution is performed over
      #                 'D', 'H', and 'W' dimensions.
      # +in_channels+:: (integer, default 0)
      #                 The number of input channels to this layer. If
      #                 not specified, initialization will be deferred
      #                 to the first time #forward is called and
      #                 +in_channels+ will be inferred from the shape
      #                 of the input data.
      # +use_bias+::    (boolean, default +true+)
      #                 Whether the layer uses a bias vector.
      # +activation+::  (string, optional)
      #                 Activation function to use. If nothing is
      #                 specified, no activation is applied (it acts
      #                 like "linear" activation: <tt>a(x) = x</tt>).
      # +prefix+::      (string, optional)
      #                 Name space for child blocks.
      # +params+::      (ParameterDict, optional)
      #                 ParameterDict for sharing weights.
      #
      def initialize(channels:, kernel_size:, strides:, padding:, dilation:,
                     layout:, in_channels: 0, use_bias: true, activation: nil,
                     op_name: 'Convolution', prefix: nil, params: nil)
        super(prefix: prefix, params: params)
        self.with_name_scope do
          @channels = channels
          @in_channels = in_channels
          if strides.is_a?(Numeric)
            strides = [strides] * kernel_size.length
          end
          if padding.is_a?(Numeric)
            padding = [padding] * kernel_size.length
          end
          if dilation.is_a?(Numeric)
            dilation = [dilation] * kernel_size.length
          end
          @op_name = op_name
          @kwargs = {
            kernel: kernel_size,
            stride: strides,
            pad: padding,
            dilate: dilation,
            no_bias: !use_bias,
            num_filter: channels,
            layout: layout
          }
          shape = [0] * (kernel_size.length + 2)
          shape[layout.index('C')] = in_channels
          shape[layout.index('N')] = 1
          _, wshape, bshape = infer_weight_shape(@op_name, shape, @kwargs)
          self[:weight] = self.params.get(
            'weight',
            shape: wshape,
            allow_deferred_init: true
          )
          if use_bias
            self[:bias] = self.params.get(
              'bias',
              shape: bshape,
              allow_deferred_init: true
            )
          else
            self[:bias] = nil
          end
          if activation
            self[:act] = MXNet::Gluon::NN.Activation(
              activation,
              prefix: "#{activation}_"
            )
          else
            self[:act] = nil
          end
        end
      end

      def hybrid_forward(clazz, data, weight = nil, bias = nil, **kwargs)
        weight ||= kwargs[:weight]
        bias ||= kwargs[:bias]
        if bias.nil?
          out = clazz.send(@op_name, data, weight, **@kwargs)
        else
          out = clazz.send(@op_name, data, weight, bias, **@kwargs)
        end
        if self[:act]
          out = self[:act].(out)
        end
        out
      end

      private

      def infer_weight_shape(op_name, shape, **kwargs)
        sym = MXNet::Symbol.var('data', shape: shape)
        sym = MXNet::Symbol.send(op_name, sym, **kwargs)
        sym.infer_shape_partial[0]
      end

      def hint
        'conv'
      end
    end

    ##
    # Base class for pooling layers.
    #
    class Pooling < MXNet::Gluon::HybridBlock
      ##
      # Creates a new instance.
      #
      # ====Parameters
      #
      # +pool_size+::   (array of N integers)
      #                 Specifies the dimensions of pooling
      #                 operation.
      # +strides+::     (integer or array of N integers)
      #                 Specifies the strides of the pooling
      #                 operation.
      # +padding+::     (integer or array of N integers)
      #                 If padding is non-zero, then the input is
      #                 implicitly zero-padded on both sides for
      #                 +padding+ number of points.
      # +prefix+::      (string, optional)
      #                 Name space for child blocks.
      # +params+::      (ParameterDict, optional)
      #                 ParameterDict for sharing weights.
      #
      def initialize(pool_size:, strides:, padding:,
                     prefix: nil, params: nil)
        super(prefix: prefix, params: params)
        if strides.nil?
          strides = pool_size
        end
        if strides.is_a?(Numeric)
          strides = [strides] * pool_size.length
        end
        if padding.is_a?(Numeric)
          padding = [padding] * pool_size.length
        end
        @kwargs = {
          kernel: pool_size,
          stride: strides,
          pad: padding
        }
      end

      def hybrid_forward(clazz, data)
        clazz.Pooling(data, **@kwargs)
      end

      private

      def hint
        'pool'
      end
    end
  end

  ##
  # 2D convolution layer (e.g. spatial convolution over images).
  #
  # This layer creates a convolution kernel that is convolved with the
  # input to produce a tensor of outputs. If +use_bias+ is +true+, a
  # bias vector is created and added to the outputs.  Finally, if
  # +activation+ is not +nil+, the activation is applied to the
  # outputs. If +in_channels+ is not specified, parameter
  # initialization will be deferred to the first time #forward is
  # called and +in_channels+ will be inferred from the shape of input
  # data.
  #
  class Conv2D < MXNet::Gluon::NN::Internal::Conv
    ##
    # Creates a new instance.
    #
    # ====Parameters
    #
    # +channels+::    (integer)
    #                 The dimensionality of the output space (the
    #                 number of output channels in the convolution).
    # +kernel_size+:: (integer or array of 2 integers)
    #                 Specifies the size of the convolution window.
    # +strides+::     (integer or array of 2 integers, default 1)
    #                 Specifies the strides of the convolution.
    # +padding+::     (integer or array of 2 integers, default 0)
    #                 If padding is non-zero, then the input is
    #                 implicitly zero-padded on both sides for
    #                 +padding+ number of points.
    # +dilation+::    (integer or array of 2 integers, default 1)
    #                 Specifies the dilation rate to use for dilated
    #                 convolution.
    # +layout+::      (string, default 'NCHW')
    #                 Dimension ordering of data and weight. Only
    #                 supports 'NCHW' and 'NHWC' layout for now. 'N',
    #                 'C', 'H', 'W' stands for batch, channel, height,
    #                 and width dimensions respectively. Convolution
    #                 is applied on the 'H' and 'W' dimensions.
    # +in_channels+:: (integer, default 0)
    #                 The number of input channels to this layer. If
    #                 not specified, initialization will be deferred
    #                 to the first time #forward is called and
    #                 +in_channels+ will be inferred from the shape
    #                 of the input data.
    # +use_bias+::    (boolean, default +true+)
    #                 Whether the layer uses a bias vector.
    # +activation+::  (string, optional)
    #                 Activation function to use. If nothing is
    #                 specified, no activation is applied (it acts
    #                 like "linear" activation: <tt>a(x) = x</tt>).
    # +prefix+::      (string, optional)
    #                 Name space for child blocks.
    # +params+::      (ParameterDict, optional)
    #                 ParameterDict for sharing weights.
    #
    def initialize(channels:, kernel_size:, strides: 1, padding: 0, dilation: 1,
                   layout: 'NCHW', in_channels: 0, use_bias: true, activation: nil,
                   prefix: nil, params: nil)
      if kernel_size.is_a?(Numeric)
        kernel_size = [kernel_size] * 2
      end
      unless kernel_size.length == 2
        raise ArgumentError, "kernel_size must be an integer or an array of 2 integers"
      end
      super(
        channels: channels,
        kernel_size: kernel_size,
        strides: strides,
        padding: padding,
        dilation: dilation,
        layout: layout,
        in_channels: in_channels,
        use_bias: use_bias,
        activation: activation,
        prefix: prefix,
        params: params
      )
    end
  end

  def self.Conv2D(*args)
    Conv2D.new(*args)
  end

  ##
  # Max pooling operation for 2D data (e.g. images).
  #
  class MaxPool2D < MXNet::Gluon::NN::Internal::Pooling
    ##
    # Creates a new instance.
    #
    # ====Parameters
    #
    # +pool_size+::   (integer or array of 2 integers, default 2)
    #                 Specifies the size of pooling window.
    # +strides+::     (integer or array of 2 integers, default +nil+)
    #                 Specifies the strides of the pooling operation.
    # +padding+::     (integer or array of 2 integers, default 0)
    #                 If padding is non-zero, then the input is
    #                 implicitly zero-padded on both sides for
    #                 +padding+ number of points.
    # +prefix+::      (string, optional)
    #                 Name space for child blocks.
    # +params+::      (ParameterDict, optional)
    #                 ParameterDict for sharing weights.
    #
    def initialize(pool_size: 2, strides: nil, padding: 0,
                   prefix: nil, params: nil)
      if pool_size.is_a?(Numeric)
        pool_size = [pool_size] * 2
      end
      unless pool_size.length == 2
        raise ArgumentError, "pool_size must be an integer or an array of 2 integers"
      end
      super(
        pool_size: pool_size,
        strides: strides,
        padding: padding,
        prefix: prefix,
        params: params
      )
    end
  end

  def self.MaxPool2D(*args)
    MaxPool2D.new(*args)
  end

  ##
  # Flattens the input to two dimensions.
  #
  # ====Inputs
  #
  # Tensor with arbitrary shape: <tt>[N, x1, x2, ..., xn]</tt>
  #
  # ====Output
  #
  # Tensor with shape: <tt>[N, x1 * x2 * ... * xn]</tt>
  #
  class Flatten < MXNet::Gluon::HybridBlock
    ##
    # Creates a new instance.
    #
    def initialize(**kwargs)
      super(**kwargs)
    end

    def hybrid_forward(clazz, data)
      clazz.flatten(data)
    end
  end

  def self.Flatten(*args)
    Flatten.new(*args)
  end
end
