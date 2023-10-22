# Integrating Singular Spectrum Analysis with Deep Learning Algorithms for Australian Stock Forecast

The project aims to implement SSA algori

## Description

This paper proposes a model combining Singular Spectrum Analysis (SA) and Deep Learning (DL), targeting stocks on ASX 50 in the Australian market. First, stock prices are decomposed into valuable sequences and eliminating noise, and then multiple DLs are trained using the denoised series to make forecasts and construct appropriate trading strategies. The experimental results show that the SSA-DL models can uncover valid information in stock prices, construct denoised stock price prediction data, and obtain better prediction results as well as investment returns. The best model obtained in this study was SSA-CNN-LSTM, which was able to generate a Sharpe Ratio of up to 1.88 and 67% ROI.

## Getting Started

### Dataset description (Data repository)

* All the original data of stocks price are extracted from [yahoo finance](https://finance.yahoo.com/), dates from 1/2/2018 until 3/31/2023. And the data are stored in ASX_stockconsidering repository.
* The Grouped repository contains data after grouping according to sectors, and also data eliminated based on start time.
* Generated repository contains dataset after processing from SSA algorithm.
* Prediction repository only contains prediction result from CNN model.
* Parameter tuning repositories contain results from all the other models with different hyper parameters.
* MAE_MSE repository contains evaluation metrics.

# Models' Code
* Files to gerenate available stock data
* Files to implement SSA algorithm
* Files to implement different deep learning algorithms
* Files to evaluate the models' performance

### Trading Strategy Repository
* After obtaining all the results from the models' prediction, these codes are run to provide trading startegy results and financial metrics for further domain evaluation.

### Installing

```
pip install -r requirements.txt
```

### Executing program

Simply open every ipynb file and run them with the right kernel.

## Authors

Contributors names and contact info

[Fan Zixian](http://linkedin.com/in/zixian-demitry-fan-607611212): u3577161@connect.hku.hk

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
