
"""unemployment_analysis.py
Load 'Unemployment in India.csv' from the same folder (/mnt/data),
clean and explore the unemployment rate, visualize trends,
analyze COVID impact, and save outputs (plots and summary CSVs).
Usage: python unemployment_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), 'Unemployment in India.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean(path=DATA_PATH):
    # Load with fallback encoding
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding='latin1')
    # Strip column names
    df.columns = [c.strip() for c in df.columns]
    # Heuristically pick date and rate columns
    date_col = None
    rate_col = None
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in ['date','year','month']):
            date_col = date_col or c
        if any(k in low for k in ['unemploy','unemployment','unemployed','rate','%']):
            rate_col = rate_col or c
    if date_col is None:
        date_col = df.columns[0]
    if rate_col is None:
        # choose first numeric column or second column
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        rate_col = numeric_cols[0] if numeric_cols else df.columns[1]
    # Build cleaned df
    clean = df[[date_col, rate_col]].copy()
    clean.columns = ['Date', 'UnemploymentRate']
    # Clean rate values
    clean['UnemploymentRate'] = clean['UnemploymentRate'].astype(str).str.replace('%','').str.replace(',','').str.strip()
    clean['UnemploymentRate'] = pd.to_numeric(clean['UnemploymentRate'], errors='coerce')
    # Parse dates with flexible formats
    clean['Date'] = pd.to_datetime(clean['Date'], errors='coerce', dayfirst=False)
    if clean['Date'].isna().sum() > len(clean)*0.2:
        clean['Date2'] = pd.to_datetime(clean['Date'].astype(str).str.replace(' ', '-'), errors='coerce', dayfirst=False)
        clean['Date'] = clean['Date'].fillna(clean['Date2'])
        clean.drop(columns=['Date2'], inplace=True)
    clean = clean.dropna(subset=['Date','UnemploymentRate']).sort_values('Date').reset_index(drop=True)
    clean['Date'] = pd.to_datetime(clean['Date'])
    clean = clean.set_index('Date')
    return clean

def save_summary_tables(df, outdir=OUTPUT_DIR):
    # Summary stats
    summary = df['UnemploymentRate'].describe().to_frame().T
    summary.to_csv(os.path.join(outdir, 'summary_stats.csv'), index=False)
    # Yearly and monthly averages
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    yearly = df.groupby('Year')['UnemploymentRate'].mean().reset_index()
    monthly = df.groupby('Month')['UnemploymentRate'].mean().reset_index()
    yearly.to_csv(os.path.join(outdir, 'yearly_average.csv'), index=False)
    monthly.to_csv(os.path.join(outdir, 'monthly_average.csv'), index=False)
    return summary, yearly, monthly

def plot_timeseries(df, outdir=OUTPUT_DIR):
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['UnemploymentRate'], marker='o', linestyle='-')
    plt.title('Unemployment Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'timeseries_unemployment.png'), dpi=150)
    plt.close()

def plot_with_rolling(df, outdir=OUTPUT_DIR, window=3):
    df['Rolling'] = df['UnemploymentRate'].rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['UnemploymentRate'], label='Rate', marker='o')
    plt.plot(df.index, df['Rolling'], label=f'{window}-period rolling mean', linestyle='--')
    plt.title('Unemployment Rate with Rolling Mean')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rolling_unemployment.png'), dpi=150)
    plt.close()

def plot_yearly(yearly, outdir=OUTPUT_DIR):
    plt.figure(figsize=(8,4))
    plt.plot(yearly['Year'], yearly['UnemploymentRate'], marker='o')
    plt.title('Yearly Average Unemployment Rate')
    plt.xlabel('Year')
    plt.ylabel('Average Unemployment Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'yearly_average.png'), dpi=150)
    plt.close()

def plot_monthly(monthly, outdir=OUTPUT_DIR):
    plt.figure(figsize=(8,4))
    plt.plot(monthly['Month'], monthly['UnemploymentRate'], marker='o')
    plt.title('Average Unemployment Rate by Month')
    plt.xlabel('Month (1=Jan)')
    plt.ylabel('Average Unemployment Rate (%)')
    plt.xticks(range(1,13))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'monthly_average.png'), dpi=150)
    plt.close()

def covid_impact(df, outdir=OUTPUT_DIR):
    pre_covid_end = pd.to_datetime('2020-02-29')
    covid_end = pd.to_datetime('2021-12-31')
    pre = df[df.index <= pre_covid_end]['UnemploymentRate']
    during = df[(df.index > pre_covid_end) & (df.index <= covid_end)]['UnemploymentRate']
    post = df[df.index > covid_end]['UnemploymentRate']
    rows = [
        ('Pre-COVID (<=2020-02)', pre.count(), pre.mean(), pre.median(), pre.max()),
        ('COVID (2020-03 to 2021-12)', during.count(), during.mean(), during.median(), during.max()),
        ('Post-COVID (2022+)', post.count(), post.mean(), post.median(), post.max())
    ]
    covid_df = pd.DataFrame(rows, columns=['Period','Count','Mean','Median','Max'])
    covid_df.to_csv(os.path.join(outdir, 'covid_period_comparison.csv'), index=False)
    # Simple percent changes
    pre_mean = pre.mean()
    during_mean = during.mean() if during.count()>0 else np.nan
    during_peak = during.max() if during.count()>0 else np.nan
    pct_mean = (during_mean - pre_mean)/pre_mean*100 if pre_mean and during_mean else np.nan
    pct_peak = (during_peak - pre_mean)/pre_mean*100 if pre_mean and during_peak else np.nan
    with open(os.path.join(outdir, 'covid_insights.txt'), 'w') as f:
        f.write(f'Pre-COVID mean: {pre_mean}\n')
        f.write(f'COVID mean: {during_mean}\n')
        f.write(f'COVID peak: {during_peak}\n')
        f.write(f'Pct increase in mean: {pct_mean}\n')
        f.write(f'Pct increase in peak: {pct_peak}\n')
    return covid_df

def main():
    print('Loading and cleaning data...')
    df = load_and_clean()
    print('Saving summary tables...')
    summary, yearly, monthly = save_summary_tables(df)
    print('Creating plots...')
    plot_timeseries(df)
    plot_with_rolling(df)
    plot_yearly(yearly)
    plot_monthly(monthly)
    print('Analyzing COVID impact...')
    covid_df = covid_impact(df)
    print('All outputs saved in the output/ folder alongside the CSV.')

if __name__ == '__main__':
    main()
