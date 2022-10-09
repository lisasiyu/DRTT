# STEP 1 Training M-MLM and T-MLM
sh ./myscript/train_mlm.sh
sh ./myscript/train_tlm.sh

# STEP 2 Training forward and backward baseline models
sh ./myscript/train_baseline.sh

# STEP 3 Generating adversarial examples
sh ./myscript/multi_data_aug.sh

# STEP 4 Filtering with our definition
sh ./myscript/filter.sh

# STEP 5 Training DRTT model
sh ./myscript/train_chen.sh

# STEP 6 Testing on noisy testset
sh ./myscript/test_noisy.sh
