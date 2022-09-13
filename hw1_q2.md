PM616 HW1 Q2
================
Sylvia Shen
2022-09-12

*Simulate a small dataset (with n = 1000) with a binary outcome. Fit a
logistic regression model and an ANN model without any hidden layer.
Compare the results.*

``` r
set.seed(1)
sample_size = 1000
x1 = rnorm(sample_size, mean = 0, sd = 1)
x2 = rnorm(sample_size, mean = 0, sd = 1)
beta1 = 1
beta2 = 2
eta = beta1*x1 + beta2*x2
linear_pred = 1/(1+exp(-eta))
y = rbinom(sample_size, size = 1, prob = linear_pred)
```

### Logistic Regression

``` r
model.glm = glm(y~x1+x2, family = "binomial")
summary(model.glm)
```

    ## 
    ## Call:
    ## glm(formula = y ~ x1 + x2, family = "binomial")
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5761  -0.6817   0.1017   0.6875   2.2861  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  0.13552    0.08405   1.612    0.107    
    ## x1           0.95310    0.09617   9.911   <2e-16 ***
    ## x2           1.93298    0.12765  15.143   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1385.51  on 999  degrees of freedom
    ## Residual deviance:  873.57  on 997  degrees of freedom
    ## AIC: 879.57
    ## 
    ## Number of Fisher Scoring iterations: 5

### ANN

``` r
model.ann <- keras_model_sequential() %>% 
  layer_dense(1, activation = "sigmoid")
```

    ## Loaded Tensorflow version 2.9.2

``` r
model.ann %>% compile(
  optimizer='adam', 
  loss='binary_crossentropy', 
  metrics='accuracy'
)

model.ann.fit = model.ann %>% fit(as.matrix(cbind(x1,x2)), 
                          as.matrix(y),
                          # validation_split = 0.2,
                          epochs = 200, batch_size = 50
                             )

summary(model.ann)
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ##  Layer (type)                       Output Shape                    Param #     
    ## ================================================================================
    ##  dense (Dense)                      (50, 1)                         3           
    ## ================================================================================
    ## Total params: 3
    ## Trainable params: 3
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

``` r
ann.weights = get_weights(model.ann)
```

### Comparison

``` r
data.frame(parameter = c("bias", "x1", "x2"), 
           truth = c(0,beta1, beta2),
           logistic = as.numeric(coef(model.glm)), 
           ANN = c(ann.weights[[2]], ann.weights[[1]][,1])) %>% 
  knitr::kable(digits = 3)
```

| parameter | truth | logistic |   ANN |
|:----------|------:|---------:|------:|
| bias      |     0 |    0.136 | 0.112 |
| x1        |     1 |    0.953 | 0.785 |
| x2        |     2 |    1.933 | 1.487 |

In this simulation, the parameter estimates from the ANN model is very
close to those from fitting a GLM. By specifying `epoch = 200` and
`batch_size = 50`, the loss stabilized after epoch \#150 at around 0.47.
But the GLM estimates are still slightly less biased compared to the ANN
estimates. This might be due to the fact that the iteratively reweighted
least squares (IRLS) algorithm used in `glm()` is more robust in this
case than Stochastic Gradient Descent in the ANN.
