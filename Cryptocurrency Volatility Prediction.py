import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import mplfinance as mpf
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from arch.unitroot import ADF
import pickle
import warnings
warnings.filterwarnings('ignore')


class CryptoVolatilityAnalyzer:
    def __init__(self, path):
        """
        Initialize the analyzer with the path to the data.

        Parameters:
        - path (str): Directory path containing cryptocurrency CSV files.
        """
        self.path = path
        self.dataframes = {}
        self.vol = {}
        self.model_stats = {}
        self.stats_dict = {}
        self.garch_predictions = {}
        self.tgarch_predictions = {}
        self.egarch_predictions = {}
        self.convergence_summary = {
            'GARCH': {'Converged': 0, 'Failed': 0},
            'TGARCH': {'Converged': 0, 'Failed': 0},
            'EGARCH': {'Converged': 0, 'Failed': 0}
        }
        self.garch_results = {}
        self.tgarch_results = {}
        self.egarch_results = {}

    def load_and_preprocess_data(self):
        """
        Load and preprocess cryptocurrency data from CSV files in the specified directory.
        """
        for file in os.listdir(self.path):
            file_path = os.path.join(self.path, file)
            if file.lower() == 'current crypto leaderboard.csv':
                continue  # Skip irrelevant file, since this csv is not a data file of crypytocurrency.
            else:
                try:
                    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                    df_name = file[:-4].lower()  # Normalize name to lowercase

                    if df_name not in self.model_stats:
                        self.model_stats[df_name] = {
                            'GARCH AIC': np.nan,
                            'GARCH BIC': np.nan,
                            'TGARCH AIC': np.nan,
                            'TGARCH BIC': np.nan,
                            'EGARCH AIC': np.nan,
                            'EGARCH BIC': np.nan,
                            'JB Statistic': np.nan,
                            'JB P-Value': np.nan,
                            'AD Statistic': np.nan,
                            'AD P-Value': np.nan
                        }

                    # Convert 'Close' to numeric, coerce errors to NaN
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

                    # Replace zeros with NaN to avoid infinite returns
                    df['Close'].replace(0, np.nan, inplace=True)

                    df['Close'].fillna(method='ffill', inplace=True) # using forward fill is better than filling NaN with mean/mode or linear interpolation.
                    df.dropna(subset=['Close'], inplace=True)

                    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                    df['Returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
                    df.dropna(subset=['Returns'], inplace=True)

                    self.dataframes[df_name] = df


                    returns = df['Returns']
                    volatility = returns.std() * np.sqrt(365)
                    self.vol[df_name] = volatility

                    # Normality Tests
                    returns_data = df['Returns'].tail(100)
                    jb_stat, p_value_JB = jarque_bera(returns_data)
                    ad_statistic, p_value_AD = normal_ad(returns_data)
                    self.model_stats[df_name].update({
                        'JB Statistic': jb_stat,
                        'JB P-Value': p_value_JB,
                        'AD Statistic': ad_statistic,
                        'AD P-Value': p_value_AD
                    })

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    def calculate_descriptive_statistics(self):
        """
        Calculate descriptive statistics for each DataFrame and print ranked lists for volatility and close prices.
        
        Volatility is defined as the standard deviation of the 'Returns' column.
        Close Price ranking is based on the latest available 'Close' price.
        """
        # Initialize the stats dictionary if not already present
        if not hasattr(self, 'stats_dict'):
            self.stats_dict = {}
        
        volatility_dict = {}
        latest_close_dict = {}
        skewness_dict = {}
        kurtosis_dict = {}
        
        for df_name, df in self.dataframes.items():
            # Ensure 'Returns' and 'Close' columns exist and are numeric
            required_columns = ['Returns', 'Close']
            for col in required_columns:
                if col not in df.columns:
                    print(f"'{col}' column not found in dataframe for {df_name}. Skipping.")
                    continue
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN in 'Returns' or 'Close'
            df_clean = df.dropna(subset=required_columns)
            
            if df_clean.empty:
                print(f"No valid data in dataframe for {df_name} after cleaning. Skipping.")
                continue
            
            # Calculate descriptive statistics
            stats_df = df_clean.describe()
            self.stats_dict[df_name] = stats_df
            
            # Calculate skewness and kurtosis
            skewness = df_clean['Returns'].skew()
            kurtosis = df_clean['Returns'].kurtosis()
            skewness_dict[df_name] = skewness
            kurtosis_dict[df_name] = kurtosis
            
            # Calculate volatility as the standard deviation of 'Returns'
            volatility = df_clean['Returns'].std()
            volatility_dict[df_name] = volatility
            
            # Get the latest 'Close' price
            latest_close = df_clean['Close'].iloc[-1]
            latest_close_dict[df_name] = latest_close
        
        if not self.stats_dict:
            print("No descriptive statistics were calculated. Please check your data.")
            return
        
        # Create DataFrames for volatility, skewness, kurtosis, and latest close prices
        volatility_series = pd.Series(volatility_dict, name='Volatility (Std Dev)')
        latest_close_series = pd.Series(latest_close_dict, name='Latest Close Price')
        skewness_series = pd.Series(skewness_dict, name='Skewness')
        kurtosis_series = pd.Series(kurtosis_dict, name='Kurtosis')
        
        # Rank cryptocurrencies by Volatility (descending: higher volatility is riskier)
        ranked_volatility = volatility_series.sort_values(ascending=False)
        
        # Rank cryptocurrencies by Latest Close Price (descending: higher price ranks higher)
        ranked_close = latest_close_series.sort_values(ascending=False)
        
        # Rank by Skewness and Kurtosis
        ranked_skewness = skewness_series.sort_values(ascending=False)
        ranked_kurtosis = kurtosis_series.sort_values(ascending=False)
        
        # Print Ranked Volatility
        print("\nRanked Volatility (Standard Deviation of Returns):")
        print("------------------------------------------------")
        print(ranked_volatility.to_string())
        
        # Print Ranked Close Prices
        print("\nRanked Latest Close Prices:")
        print("----------------------------")
        print(ranked_close.to_string())
        
        # Print Ranked Skewness
        print("\nRanked Skewness of Returns:")
        print("----------------------------")
        print(ranked_skewness.to_string())
        
        # Print Ranked Kurtosis
        print("\nRanked Kurtosis of Returns:")
        print("----------------------------")
        print(ranked_kurtosis.to_string())
        
        print("\nDescriptive statistics and rankings have been calculated and displayed.")

    def print_all_statistics(self):
        """
        Print descriptive statistics for all cryptocurrencies.
        """
        if not self.stats_dict:
            print("Descriptive statistics not calculated yet. Please run calculate_descriptive_statistics() first.")
            return

        for df_name, stats in self.stats_dict.items():
            print(f"\nDescriptive Statistics for {df_name.capitalize()}:")
            print(stats)
            print("-" * 60)

    def print_specific_statistics(self, crypto_name):
        """
        Print descriptive statistics for a specific cryptocurrency.

        Parameters:
        - crypto_name (str): Name of the cryptocurrency (case-insensitive).
        """
        crypto_name = crypto_name.lower()
        if crypto_name not in self.stats_dict:
            if crypto_name in self.dataframes:
                print("Descriptive statistics not calculated yet. Calculating now...")
                self.calculate_descriptive_statistics()
            else:
                print(f"Cryptocurrency '{crypto_name}' not found.")
                return

        stats = self.stats_dict.get(crypto_name)
        if stats is not None:
            print(f"\nDescriptive Statistics for {crypto_name.capitalize()}:")
            print(stats)
            print("-" * 60)
        else:
            print(f"Descriptive statistics for '{crypto_name}' not available.")

    def plot_rolling_statistics(self, df_name, window=100):
        """
        Plot rolling mean and rolling volatility for a given cryptocurrency.
        
        Parameters:
        - df_name (str): Name of the cryptocurrency.
        - window (int): Window size for rolling calculations.
        """
        if df_name not in self.dataframes:
            print(f"{df_name} not found in dataframes.")
            return

        df = self.dataframes[df_name]
        rolling_mean = df['Returns'].rolling(window=window).mean()
        rolling_vol = df['Returns'].rolling(window=window).std()

        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=120)

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Rolling Mean', color=color)
        ax1.plot(df.index, rolling_mean, color=color, label='Rolling Mean')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Rolling Volatility', color=color)  # We already handled the x-label with ax1
        ax2.plot(df.index, rolling_vol, color=color, label='Rolling Volatility')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title(f'Rolling Mean and Volatility for {df_name.capitalize()} (Window={window})')
        plt.tight_layout()
        plt.show()

    def calculate_max_drawdown(self):
        """
        Calculate and display the maximum drawdown for each cryptocurrency.
        """
        max_drawdowns = {}
        for df_name, df in self.dataframes.items():
            close = df['Close']
            rolling_max = close.cummax()
            drawdown = (close - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            max_drawdowns[df_name] = max_drawdown
        
        # Create a DataFrame and sort
        max_drawdown_df = pd.DataFrame(list(max_drawdowns.items()), columns=['Cryptocurrency', 'Max Drawdown'])
        max_drawdown_df['Max Drawdown'] = max_drawdown_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
        max_drawdown_df.sort_values(by='Max Drawdown', inplace=True)
        max_drawdown_df.reset_index(drop=True, inplace=True)
        
        print("\nMaximum Drawdown for Each Cryptocurrency:")
        print("-----------------------------------------")
        print(max_drawdown_df)
        
        return max_drawdown_df
    
    def analyze_trading_volume(self):
        """
        Analyze and plot trading volumes for each cryptocurrency.
        """
        for df_name, df in self.dataframes.items():
            if 'Volume' not in df.columns:
                print(f"'Volume' column not found for {df_name}. Skipping.")
                continue
            
            plt.figure(figsize=(12, 6), dpi=120)
            plt.plot(df.index, df['Volume'], color='purple')
            plt.title(f'Trading Volume for {df_name.capitalize()}')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.grid(True)
            plt.show()

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """
        Calculate and display the Sharpe Ratio for each cryptocurrency.
        
        Parameters:
        - risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation.
        """
        sharpe_ratios = {}
        for df_name, df in self.dataframes.items():
            returns = df['Returns']
            excess_returns = returns - risk_free_rate / 365  # Assuming daily returns
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(365)
            sharpe_ratios[df_name] = sharpe_ratio
        
        # Create a DataFrame and sort
        sharpe_df = pd.DataFrame(list(sharpe_ratios.items()), columns=['Cryptocurrency', 'Sharpe Ratio'])
        sharpe_df.sort_values(by='Sharpe Ratio', ascending=False, inplace=True)
        sharpe_df.reset_index(drop=True, inplace=True)
        
        print("\nSharpe Ratio for Each Cryptocurrency:")
        print("--------------------------------------")
        print(sharpe_df)
        
        return sharpe_df

    def residual_analysis(self, df_name, model_type):
        """
        Perform residual analysis for a given cryptocurrency and model type.

        Parameters:
        - df_name (str): Name of the cryptocurrency.
        - model_type (str): Type of the model ('GARCH', 'TGARCH', 'EGARCH').
        """
        # Retrieve the appropriate model results
        model_type = model_type.upper()
        if model_type == 'GARCH':
            res = self.garch_results.get(df_name)
        elif model_type == 'TGARCH':
            res = self.tgarch_results.get(df_name)
        elif model_type == 'EGARCH':
            res = self.egarch_results.get(df_name)
        else:
            print(f"Unsupported model type: {model_type}")
            return

        if res is None:
            print(f"No fitted {model_type} model found for {df_name}. Please fit the model first.")
            return

        # Standardized residuals
        standardized_residuals = res.resid / res.conditional_volatility

        # Q-Q Plot
        sm.qqplot(standardized_residuals, line='s')
        plt.title(f'Q-Q Plot of Standardized Residuals for {df_name.capitalize()} ({model_type})')
        plt.show()

        # Ljung-Box Test
        lb_test = acorr_ljungbox(standardized_residuals, lags=[10], return_df=True)
        print(f"Ljung-Box Test for {df_name.capitalize()} ({model_type}):")
        print(lb_test)

        # AD Test
        ad_statistic, p_value_AD = normal_ad(standardized_residuals)
        print(f"AD Test for {df_name.capitalize()} ({model_type}):")
        print(f"AD Statistic: {ad_statistic}, AD P-Value: {p_value_AD}")

        # Jarque-Bera Test
        jb_stat, jb_pvalue = jarque_bera(standardized_residuals)
        print(f"Jarque-Bera Test for {df_name.capitalize()} ({model_type}):")
        print(f"JB Statistic: {jb_stat}, JB P-Value: {jb_pvalue}")
        
        # ADF Test
        adf_test = ADF(standardized_residuals)
        adf_stat, adf_pvalue = adf_test.stat, adf_test.pvalue
        print(f"ADF Test for {df_name.capitalize()} ({model_type}):")
        print(f"ADF Statistic: {adf_stat}, ADF P-Value: {adf_pvalue}")
    
    # cryptocurrencies often exhibit heavy tails and skewness, start trying with StudentsT for dist would be appropriate, adjust according to QQ-plot and other statistical results.
    # dist: 'normal' | 'gaussian' | 't' | 'studentst' | 'skewstudent' | 'skewt' | 'ged' | 'generalized error' = 'normal'
    def fit_garch(self, p=1, q=1, dist='studentst', rescale=True):
        """
        Fit a GARCH(1,1) model to all cryptocurrencies.

        Parameters:
        - p (int): Order of the GARCH model.
        - q (int): Order of the GARCH model.
        - dist (str): Distribution for the error term.
        - rescale (bool): Whether to rescale the data.

        Updates:
        - self.garch_predictions
        - self.model_stats
        - self.convergence_summary
        - self.garch_results
        """
        for df_name, df in self.dataframes.items():
            returns = df['Returns']
            try:
                model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist, rescale=rescale)
                res = model.fit(update_freq=5, disp='off')

                if res.convergence_flag == 0:
                    last_vol = res.conditional_volatility.iloc[-1] * np.sqrt(365)
                    self.garch_predictions[df_name] = last_vol
                    self.model_stats[df_name]['GARCH AIC'] = res.aic
                    self.model_stats[df_name]['GARCH BIC'] = res.bic
                    self.convergence_summary['GARCH']['Converged'] += 1
                    # Store the fitted model result
                    self.garch_results[df_name] = res
                else:
                    self.model_stats[df_name]['GARCH AIC'] = np.nan
                    self.model_stats[df_name]['GARCH BIC'] = np.nan
                    self.convergence_summary['GARCH']['Failed'] += 1

            except Exception as e:
                print(f"GARCH model fitting failed for {df_name}: {e}")
                self.model_stats[df_name]['GARCH AIC'] = np.nan
                self.model_stats[df_name]['GARCH BIC'] = np.nan
                self.convergence_summary['GARCH']['Failed'] += 1

        print("GARCH model fitting completed.")

    def fit_tgarch(self, p=1, q=1, o=1, dist='studentst', rescale=True):
        """
        Fit a TGARCH(1,1,1) model to all cryptocurrencies.

        Parameters:
        - p (int): Order of the GARCH model.
        - q (int): Order of the GARCH model.
        - o (int): Order of the asymmetric term.
        - dist (str): Distribution for the error term.
        - rescale (bool): Whether to rescale the data.

        Updates:
        - self.tgarch_predictions
        - self.model_stats
        - self.convergence_summary
        - self.tgarch_results
        """
        for df_name, df in self.dataframes.items():
            returns = df['Returns']
            try:
                model = arch_model(returns, vol='Garch', p=p, q=q, o=o, dist=dist, rescale=rescale)
                res = model.fit(update_freq=5, disp='off')

                if res.convergence_flag == 0:
                    last_vol = res.conditional_volatility.iloc[-1] * np.sqrt(365)
                    self.tgarch_predictions[df_name] = last_vol
                    self.model_stats[df_name]['TGARCH AIC'] = res.aic
                    self.model_stats[df_name]['TGARCH BIC'] = res.bic
                    self.convergence_summary['TGARCH']['Converged'] += 1
                    # Store the fitted model result
                    self.tgarch_results[df_name] = res
                else:
                    self.model_stats[df_name]['TGARCH AIC'] = np.nan
                    self.model_stats[df_name]['TGARCH BIC'] = np.nan
                    self.convergence_summary['TGARCH']['Failed'] += 1

            except Exception as e:
                print(f"TGARCH model fitting failed for {df_name}: {e}")
                self.model_stats[df_name]['TGARCH AIC'] = np.nan
                self.model_stats[df_name]['TGARCH BIC'] = np.nan
                self.convergence_summary['TGARCH']['Failed'] += 1

        print("TGARCH model fitting completed.")

    def fit_egarch(self, p=1, q=1, dist='studentst', rescale=True, maxiter=10000, tol=1e-10):
        """
        Fit an EGARCH(1,1) model to all cryptocurrencies.

        Parameters:
        - p (int): Order of the EGARCH model.
        - q (int): Order of the EGARCH model.
        - dist (str): Distribution for the error term.
        - rescale (bool): Whether to rescale the data.
        - maxiter (int): Maximum number of iterations for the optimizer.
        - tol (float): Tolerance for the optimizer.

        Updates:
        - self.egarch_predictions
        - self.model_stats
        - self.convergence_summary
        - self.egarch_results
        """
        for df_name, df in self.dataframes.items():
            returns = df['Returns']
            try:
                model = arch_model(returns, vol='EGarch', p=p, q=q, dist=dist, rescale=rescale)
                res = model.fit(update_freq=5, disp='off', options={'maxiter': maxiter, 'tol': tol})

                if res.convergence_flag == 0:
                    last_vol = res.conditional_volatility.iloc[-1] * np.sqrt(365)
                    self.egarch_predictions[df_name] = last_vol
                    self.model_stats[df_name]['EGARCH AIC'] = res.aic
                    self.model_stats[df_name]['EGARCH BIC'] = res.bic
                    self.convergence_summary['EGARCH']['Converged'] += 1
                    # Store the fitted model result
                    self.egarch_results[df_name] = res
                else:
                    self.model_stats[df_name]['EGARCH AIC'] = np.nan
                    self.model_stats[df_name]['EGARCH BIC'] = np.nan
                    self.convergence_summary['EGARCH']['Failed'] += 1

            except Exception as e:
                print(f"EGARCH model fitting failed for {df_name}: {e}")
                self.model_stats[df_name]['EGARCH AIC'] = np.nan
                self.model_stats[df_name]['EGARCH BIC'] = np.nan
                self.convergence_summary['EGARCH']['Failed'] += 1

        print("EGARCH model fitting completed.")

    def plot_time_series_and_histogram(self, df_name):
            """
            Plot time series of Close prices and histogram of Returns for a given cryptocurrency.

            Parameters:
            - df_name (str): Name of the cryptocurrency.
            """
            if df_name not in self.dataframes:
                print(f"{df_name} not found in dataframes.")
                return

            df = self.dataframes[df_name]
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=120)

            # Time Series Plot
            axes[0].plot(df['Close'], color='blue')
            axes[0].set_title(f'Time Series of Close Prices for {df_name.capitalize()}')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Close Price')
            axes[0].grid(True)

            # Histogram of Returns
            sns.histplot(df['Returns'].dropna(), ax=axes[1], bins=50, kde=True, color='green')
            axes[1].set_title(f'Histogram of Log Returns for {df_name.capitalize()}')
            axes[1].set_xlabel('Log Returns')
            axes[1].set_ylabel('Frequency')

            plt.tight_layout()
            plt.show()

    def plot_correlogram(self, df_name, lags=40):
        """
        Plot ACF and PACF correlograms for Returns of a given cryptocurrency.

        Parameters:
        - df_name (str): Name of the cryptocurrency.
        - lags (int): Number of lags to display.
        """
        if df_name not in self.dataframes:
            print(f"{df_name} not found in dataframes.")
            return

        returns = self.dataframes[df_name]['Returns']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
        plot_acf(returns, lags=lags, ax=ax1)
        plot_pacf(returns, lags=lags, ax=ax2)
        plt.suptitle(f'Correlogram for {df_name.capitalize()} Returns')
        plt.show()

    def plot_correlogram_squared_returns(self, df_name, lags=40):
        """
        Plot ACF and PACF correlograms for squared Returns of a given cryptocurrency.

        Parameters:
        - df_name (str): Name of the cryptocurrency.
        - lags (int): Number of lags to display.
        """
        if df_name not in self.dataframes:
            print(f"{df_name} not found in dataframes.")
            return

        squared_returns = self.dataframes[df_name]['Returns'] ** 2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
        plot_acf(squared_returns, lags=lags, ax=ax1)
        plot_pacf(squared_returns, lags=lags, ax=ax2)
        plt.suptitle(f'Correlogram for Squared Returns of {df_name.capitalize()}')
        plt.show()

    def plot_candlestick(self, df_name):
        """
        Plot a candlestick chart for a given cryptocurrency.

        Parameters:
        - df_name (str): Name of the cryptocurrency.
        """
        if df_name not in self.dataframes:
            print(f"{df_name} not found in dataframes.")
            return

        df = self.dataframes[df_name]
        mpf.plot(df[['Open', 'High', 'Low', 'Close', 'Volume']], type='candle',
                 title=f'{df_name.capitalize()} Candlestick Chart', style='charles', volume=True)
        plt.show()

    def plot_heatmap(self):
        """
        Plot a heatmap of the correlation matrix of returns for all cryptocurrencies
        and print a ranked list of cryptocurrency correlations.
        """
        # Extract 'Returns' data for all cryptocurrencies
        returns_data = {df_name: df['Returns'] for df_name, df in self.dataframes.items() if 'Returns' in df.columns}
        
        if not returns_data:
            print("No 'Returns' data available in any dataframe to plot heatmap.")
            return
        
        # Combine returns into a single DataFrame
        combined_returns = pd.DataFrame(returns_data)
        
        if combined_returns.empty:
            print("Combined returns DataFrame is empty. Cannot compute correlations.")
            return
        
        # Compute the correlation matrix
        correlation_matrix = combined_returns.corr()
        
        if correlation_matrix.empty:
            print("Correlation matrix is empty. Cannot plot heatmap.")
            return

        # Print the ranked correlations
        print("\nRanked Cryptocurrency Correlations:")
        print("-" * 80)
        print(f"{'Pair':<40} {'Correlation':>20}")
        print("-" * 80)
        
        # Unstack the correlation matrix to get pairwise correlations
        corr_pairs = correlation_matrix.unstack()
        
        # Create a DataFrame from the series
        corr_pairs_df = pd.DataFrame(corr_pairs, columns=['Correlation']).reset_index()
        corr_pairs_df.columns = ['Crypto_1', 'Crypto_2', 'Correlation']
        
        # Remove self-correlations
        corr_pairs_df = corr_pairs_df[corr_pairs_df['Crypto_1'] != corr_pairs_df['Crypto_2']]
        
        # To avoid duplicate pairs (e.g., A-B and B-A), sort the crypto names and drop duplicates
        corr_pairs_df['Pair'] = corr_pairs_df.apply(lambda row: tuple(sorted([row['Crypto_1'], row['Crypto_2']])), axis=1)
        corr_pairs_df = corr_pairs_df.drop_duplicates(subset='Pair')
        
        # Sort by correlation descendingly
        sorted_corr = corr_pairs_df.sort_values(by='Correlation', ascending=False)
        
        # Display top positively correlated pairs
        top_n = 10  # You can adjust this number as needed
        print(f"\nTop {top_n} Positively Correlated Pairs:")
        print("-" * 80)
        for idx, row in sorted_corr.head(top_n).iterrows():
            pair = f"{row['Pair'][0]} - {row['Pair'][1]}"
            corr = row['Correlation']
            print(f"{pair:<40} {corr:>20.4f}")
        
        # Display top negatively correlated pairs
        print(f"\nTop {top_n} Negatively Correlated Pairs:")
        print("-" * 80)
        for idx, row in sorted_corr.tail(top_n).iterrows():
            pair = f"{row['Pair'][0]} - {row['Pair'][1]}"
            corr = row['Correlation']
            print(f"{pair:<40} {corr:>20.4f}")
        
        print("-" * 80)
        
        # Plotting the heatmap
        sns.set_theme(font_scale=0.9)
        plt.figure(figsize=(20, 12))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.3)
        plt.title('Correlation Matrix of Returns')
        plt.tight_layout()
        plt.show()
    
    def rank_volatility(self, days=100):
            """
            Rank cryptocurrencies based on their historical volatility over the last 'days' days.

            Parameters:
            - days (int): Number of recent days to consider for volatility calculation.

            Returns:
            - volatility_ranking (DataFrame): DataFrame containing cryptocurrencies and their volatilities, ranked from high to low.
            """
            volatilities = {}
            for df_name, df in self.dataframes.items():
                if 'Returns' in df.columns:
                    returns = df['Returns'].tail(days)
                    if len(returns) == days:
                        # Annualized volatility
                        volatility = returns.std() * np.sqrt(365)
                        volatilities[df_name] = volatility
                    else:
                        print(f"Not enough data for {df_name}; skipping.")
                else:
                    print(f"'Returns' column not found for {df_name}; skipping.")

            # Create a DataFrame and sort
            volatility_df = pd.DataFrame(list(volatilities.items()), columns=['Cryptocurrency', 'Volatility'])
            volatility_df.sort_values(by='Volatility', ascending=False, inplace=True)
            volatility_df.reset_index(drop=True, inplace=True)

            print(f"\nVolatility Ranking (last {days} days):")
            print(volatility_df)

            return volatility_df
    
    def plot_cumulative_returns(self, df_name):
        """
        Plot cumulative returns for a given cryptocurrency.
        
        Parameters:
        - df_name (str): Name of the cryptocurrency.
        """
        if df_name not in self.dataframes:
            print(f"{df_name} not found in dataframes.")
            return

        df = self.dataframes[df_name]
        cumulative_returns = (1 + df['Returns']).cumprod() - 1

        plt.figure(figsize=(12, 6), dpi=120)
        plt.plot(df.index, cumulative_returns, color='teal')
        plt.title(f'Cumulative Returns for {df_name.capitalize()}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.show()

    def identify_top_bottom_periods(self, df_name, top_n=5, bottom_n=5):
        """
        Identify top and bottom performing periods based on returns.
        
        Parameters:
        - df_name (str): Name of the cryptocurrency.
        - top_n (int): Number of top periods to identify.
        - bottom_n (int): Number of bottom periods to identify.
        """
        if df_name not in self.dataframes:
            print(f"{df_name} not found in dataframes.")
            return

        df = self.dataframes[df_name]
        top_periods = df['Returns'].nlargest(top_n)
        bottom_periods = df['Returns'].nsmallest(bottom_n)

        print(f"\nTop {top_n} Performing Days for {df_name.capitalize()}:")
        print(top_periods)
        
        print(f"\nBottom {bottom_n} Performing Days for {df_name.capitalize()}:")
        print(bottom_periods)

    def display_convergence_summary(self):
        """
        Display the convergence summary for all models.
        """
        print("\nConvergence Summary:")
        for model_type, results in self.convergence_summary.items():
            print(f"{model_type}: Converged: {results['Converged']}, Failed: {results['Failed']}")

    def display_model_selection_summary(self):
        """
        Perform model selection based on AIC and BIC and display the summary.
        """
        # Convert the model_stats dictionary to a DataFrame
        model_stats_df = pd.DataFrame.from_dict(self.model_stats, orient='index')

        model_selection_counters = {
            'GARCH': {'AIC': 0, 'BIC': 0},
            'TGARCH': {'AIC': 0, 'BIC': 0},
            'EGARCH': {'AIC': 0, 'BIC': 0}
        }

        # Iterate through each cryptocurrency's model stats
        for index, row in model_stats_df.iterrows():
            aic_values = row[['GARCH AIC', 'TGARCH AIC', 'EGARCH AIC']]
            bic_values = row[['GARCH BIC', 'TGARCH BIC', 'EGARCH BIC']]

            min_aic = aic_values.min(skipna=True)
            min_bic = bic_values.min(skipna=True)

            models_with_min_aic = aic_values[aic_values == min_aic].index.tolist()
            models_with_min_bic = bic_values[bic_values == min_bic].index.tolist()

            for model in models_with_min_aic:
                model_type = model.split()[0]  # Extract model type
                model_selection_counters[model_type]['AIC'] += 1

            for model in models_with_min_bic:
                model_type = model.split()[0]
                model_selection_counters[model_type]['BIC'] += 1

        print("\nModel Selection Summary:")
        for model_type, counts in model_selection_counters.items():
            print(f"{model_type}: Chosen by AIC {counts['AIC']} times, Chosen by BIC {counts['BIC']} times")

    def visualize_all(self):
        """
        Visualize data and model results for all cryptocurrencies.
        """
        for df_name in self.dataframes.keys():
            self.plot_time_series_and_histogram(df_name)
            self.plot_correlogram(df_name)
            self.plot_correlogram_squared_returns(df_name)
            self.plot_candlestick(df_name)
        self.plot_heatmap()

    # Out of sample forecasting and backtesting
    def forecast_volatility(self, df_name, model_type, steps, method, simulations):
        model_type = model_type.upper()

        df = self.dataframes.get(df_name)
        if df is None:
            print(f"Data for '{df_name}' not found.")
            return

        returns = df['Returns']
        train = returns.iloc[:-steps]
        test = returns.iloc[-steps:]

        try:
            # Fit the model on training data
            if model_type == 'GARCH':
                model = arch_model(train, vol='Garch', p=1, q=1, dist='skewstudent', rescale=True)
            elif model_type == 'TGARCH':
                model = arch_model(train, vol='Garch', p=1, o=1, q=1, dist='skewstudent', rescale=True)
            elif model_type == 'EGARCH':
                model = arch_model(train, vol='EGarch', p=1, q=1, dist='skewstudent', rescale=True)
            else:
                print(f"Unsupported model type: {model_type}")
                return
            maxiter = 10000
            tol = 1e-10
            res = model.fit(update_freq=5, disp='off', options={'maxiter': maxiter, 'tol': tol})

            # Forecast using the specified method
            if method == 'analytic':
                forecasts = res.forecast(horizon=1, method='analytic', reindex=False)
                predicted_vol = np.sqrt(forecasts.variance.values[-1, :]) * np.sqrt(365)
            elif method == 'simulation':
                forecasts = res.forecast(horizon=steps, method='simulation', simulations=simulations,reindex=False)
                predicted_vol = np.sqrt(forecasts.variance.mean(axis=0)) * np.sqrt(365)
            else:
                print(f"Unsupported forecast method: {method}")
                return

            # Calculate realized volatility
            realized_vol = test.rolling(window=2).std() * np.sqrt(365)
            realized_vol = realized_vol.dropna().values.flatten()

            # Ensure the lengths match
            min_len = min(len(predicted_vol), len(realized_vol))
            predicted_vol = predicted_vol[:min_len]
            realized_vol = realized_vol[:min_len]

            if min_len == 0:
                print(f"No overlapping data between predicted and realized volatilities for {df_name}.")
                return

            # Evaluate forecasts
            mse = mean_squared_error(realized_vol, predicted_vol)
            print(f"{model_type} Forecast MSE for {df_name.capitalize()}: {mse}")

        except Exception as e:
            print(f"Forecasting failed for {df_name} ({model_type}): {e}")

def main():
    path = '/Users/wcsmars/Desktop/Top 100 Crypto Coins'

    # Initialize the analyzer
    analyzer = CryptoVolatilityAnalyzer(path=path)

    analyzer.load_and_preprocess_data()

    # analyzer.calculate_descriptive_statistics()

    # Print descriptive statistics
    # analyzer.print_all_statistics()

    # analyzer.plot_rolling_statistics('bitcoin')

    # analyzer.calculate_max_drawdown()

    # analyzer.analyze_trading_volume()

    # analyzer.calculate_sharpe_ratio(risk_free_rate=0.04)  # Assuming a 4% annual risk-free rate

    # analyzer.plot_cumulative_returns('bitcoin')

    print("\nFitting GARCH model for all cryptocurrencies...")
    analyzer.fit_garch()

    print("\nFitting TGARCH model for all cryptocurrencies...")
    analyzer.fit_tgarch()

    print("\nFitting EGARCH model for all cryptocurrencies...")
    analyzer.fit_egarch()
    
    # Visualize data (For all cryptocurrency)
    # analyzer.visualize_all()

    # Plot for a specific cryptocurrency
    # crypto_name = 'bitcoin'
    # if crypto_name in analyzer.dataframes:
    #     analyzer.plot_time_series_and_histogram(crypto_name)
    #     analyzer.plot_correlogram(crypto_name)
    #     analyzer.plot_correlogram_squared_returns(crypto_name)
    #     analyzer.plot_candlestick(crypto_name)
    # else:
    #     print(f"{crypto_name} data not found.")

    # analyzer.plot_correlogram_squared_returns('bitcoin')
    # If the plot shows significant autocorrelation (indicating volatility clustering), GARCH is appropriate to predict volatility
    
    # Plot correlation heatmap
    # analyzer.plot_heatmap()

    # analyzer.plot_relative_volume(days=100, top_n=50, chart_type='bar')

    # Top and Bottom 5 Performing Days for each cryptocurrency
    # for crypto in analyzer.dataframes.keys():
    #     analyzer.identify_top_bottom_periods(crypto, top_n=5, bottom_n=5)

    # Check residual normality with ad test, jb test and qq-plots
    # Check heteroskedasticity with adf test, determines if a time series is stationary or contains a unit root.
    # As long as p-value > 0.05, there is no autocorrelation in the residuals.
    # analyzer.residual_analysis('bitcoin', 'EGARCH')
    
    # analyzer.forecast_volatility(df_name='bitcoin', model_type='EGARCH', steps=10, method='analytic', simulations=0)

    # Forecast using simulation method
    # analyzer.forecast_volatility(df_name='bitcoin', model_type='EGARCH', steps=10, method='simulation', simulations=1000)
    
    # Display convergence summary
    analyzer.display_convergence_summary()

    # Model selection summary based on AIC and BIC
    analyzer.display_model_selection_summary()


if __name__ == "__main__":
    main()