require 'mxnet/io/data_iter'
require 'sync'

module MXNet
  module IO
    # Performs pre-fetch for other data iterators.
    # 
    # This iterator will create another thread to perform +iter_next+ and then
    # store the data in memory.  It potentially accelerates the data read, at the
    # cost of more memory usage.
    class PrefetchingIter < DataIter
      class Event
        def initialize(initial_flag=false)
          @flag = initial_flag
          @mutex = Mutex.new
          @cv = ConditionVariable.new
        end

        def set
          @lock.synchronize do
            @flag = true
            @cv.broadcast
          end
        end

        def clear
          @lock.synchronize do
            @flag = false
          end
        end

        def wait(timeout=nil)
          @lock.synchronize do
            return true if @flag
            until @flag
              @cv.wait(@lock, timeout)
              return @flag if @flag || timeout
            end
          end
        end
      end

      class Finalizer
        def initialize
          @started = true
          @started_lock = Mutex.new
          @data_taken = nil
          @prefetch_threads = nil
        end

        attr_accessor :data_taken
        attr_accessor :prefetch_threads

        def finished?
          @started_lock.synchronize { !@started }
        end

        def call
          @started_lock.synchronize { @started = false }
          @data_taken.each(&:set)
          @prefetch_threads.each(&:join)
        end
      end

      def initialize(iters, rename_data: nil, rename_label: nil)
        super()
        @iters = Array(iters)
        unless (@n_iter = @iters.length) > 0
          raise ArgumentError, "no iterators given"
        end
        @rename_data = rename_data
        @rename_label = rename_label
        @batch_size = provide_data[0][1][0]
        @data_ready = Array.new(@n_iter) { Event.new }
        @data_taken = Array.new(@n_iter) { Event.new(true) }
        @current_batch = Array.new(@n_iter)
        @next_batch = Array.new(@n_iter)
        finalizer = Finalizer.new
        @prefetch_threads = Array.new(@n_iter) do |i|
          self.class.start_prefetch_thread(i, finalizer, @iter, @data_taken, @data_ready, @next_batch)
        end
        finalizer.data_taken = @data_taken
        finalizer.prefetch_threads = @prefetch_threads
        ObjectSpace.define_finalizer(self, finalizer)
      end

      def self.start_prefetch_thread(i, finalizer, iters, data_taken, data_ready, next_batch)
        data_iter = iters[i]
        data_taken_i = data_taken[i]
        data_ready_i = data_ready[i]
        Thread.start do
          loop do
            data_taken_i.wait
            break if finalizer.finished?
            next_batch[i] = data_iter_i.next_batch
            data_taken_i.clear
            data_ready_i.set
          end
        end
      end

      #TODO
      def provide_data
        if @rename_data.nil?
        else
        end
      end

      #TODO
      def provide_label
        if @rename_label.nil?
        else
        end
      end

      def reset
        @data_ready.each(&:wait)
        @iters.each(&:reset)
        @data_ready.each(&:clear)
        @data_taken.each(&:set)
      end

      def iter_next
        @data_ready.each(&:wait)
        if @next_batch[0].nil?
          unless @next_batch.all?(&:nil?)
            raise 'Number of entry mismatches between iterators'
          end
          false
        else
          unless _same_all?(@next_batch, :pad)
            raise 'Number of entry mismatches between iterators'
          end
          # TODO
          true
        end
      end

      private

      def _same_all?(ary, meth=nil)
        return true if ary.length <= 1
        if meth
          e0 = ary[0].send(meth)
          ary[1..-1].all? {|e| e.send(meth) == e0 }
        else
          ary[0..-2] == ary[1..-1]
        end
      end
    end
  end
end
