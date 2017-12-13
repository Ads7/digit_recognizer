**Digit classifier**

The project consists various approaches to solve the hand written digit classification problem.
The data set taken is MNIST dataset taken through tensorflow utils.


````
    digit_recognizer
        -- classifier // all algo logics
            -- knn
            -- backpropogation
            -- logistic_regression
        -- docs
        -- data // in git ignore
        -- experiments
            -- knn 
            -- backpropogation
            -- logistic_regression
        -- results // stores all csv generated from experiments
        -- utils
            -- data_processing // data downloading and initiating 
            -- math // basic mathematical tasks
````

all lib dependencies are given in req.txt

a basic running for all algos is given in main.py

data set used http://yann.lecun.com/exdb/mnist/
downloaded using tensor flow