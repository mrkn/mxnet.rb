module MXNet
  module Autograd
    def self.with_recording_state(is_record, train_mode)
      unless is_record.nil?
        prev_is_record = self.set_recording(is_record)
      end
      unless train_mode.nil?
        prev_train_mode = self.set_training(train_mode)
      end

      yield
    ensure
      if !is_record.nil? && prev_is_record != is_record
        self.set_recording(prev_is_record)
      end
      if !train_mode.nil? && prev_train_mode != train_mode
        self.set_training(prev_train_mode)
      end
    end
    private_class_method :with_recording_state

    # Make autograd recording scope in the given block.
    def self.record(train_mode=true, &block)
      with_recording_state(true, train_mode, &block)
    end

    def self.pause(train_mode=false, &block)
      with_recording_state(false, train_mode, &block)
    end

    def self.train_mode(&block)
      with_recording_state(nil, true, &block)
    end

    def self.predict_mode(&block)
      with_recording_state(nil, false, &block)
    end
  end
end
