import ds_maker
import train
import test

ds_maker.make()

print("Starting training")
train.train()

print("Starting testing")
test.test()