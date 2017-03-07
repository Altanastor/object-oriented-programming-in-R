# Statistical models

Truth be told, you won't be using object-oriented programming in most day to day R programming. Most analyses you do in R involve transformation of data, typically implemented as some sort of data flow, that is much better captured by functional programming. When you write such pipelines, you will probably be using polymorphic functions, but you rarely need to create your own classes. If you need to implement a new statistical model, however, you usually do want to create a new class.

A lot of data analysis requires that you infer parameters of interest or you build a model to predict properties of your data, but in many of those cases you don't necessarily need to know exactly how the model is constructed, how it infers parameters, or how it predicts new values. You can use the `coefficents` function to get inferred parameters from a fitted model or you can use the `predict` function to predict new values and you can use those two functions with almost any statistical model. That is because most models are implemented as classes with implementations for the generic functions `coefficients` and `predict`.

As an example of object-oriented programming in action, we can implement our own model in this chapter. We will keep it simple, so we can focus on the programming aspects and not the statistical theory, but still implement something that isn't already built into R. We will implement a version of Bayesian linear regression.

## Bayesian linear regression

