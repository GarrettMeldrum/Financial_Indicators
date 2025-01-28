import tkinter as tk
from tkinter import ttk
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def fetch_data_create_plot():
    ticker = ticker_var.get()
    period = period_var.get()
    interval = interval_var.get()
    indicator = indicator_var.get()
    
    if not ticker:
        return
    
    stock_data = yf.download(ticker, period=period, interval=interval)
    ax.clear()
    ax.plot(stock_data.index, stock_data['Close'], label=f'{ticker} Close Price')
    

    # Format the x-axis labels dynamically
    if period in ['1d','5d','1mo','3mo','6mo']: # Short scale
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')) # Month / Day
    else: # Long scale
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # Month / Year

    ax.tick_params(axis='x', rotation=45)
    canvas.draw()

# Main Tkinter window
root = tk.Tk()
root.title("Stock Price Viewer")
root.geometry("800x600")

ticker_var = tk.StringVar()
period_var = tk.StringVar(value='1mo')
interval_var = tk.StringVar(value='1d')
indicator_var = tk.StringVar()

# Options for dropdowns
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
intervals = ['1m', '2m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
indicators = ['Bollinger Bands','Average True Range Bands','Donchian Channels','Keltner Channels','Linear Regression Indicator']

# Dropdown menus frame
dropdown_frame = ttk.Frame(root)
dropdown_frame.pack()

ttk.Label(dropdown_frame, text="Select Ticker:").grid(row=0, column=0)
ticker_menu = ttk.Combobox(dropdown_frame, textvariable=ticker_var, values=tickers)
ticker_menu.grid(row=1, column=0)

ttk.Label(dropdown_frame, text="Select Time Frame:").grid(row=0, column=1)
period_menu = ttk.Combobox(dropdown_frame, textvariable=period_var, values=periods)
period_menu.grid(row=1, column=1)

ttk.Label(dropdown_frame, text="Select Interval:").grid(row=0, column=2)
interval_menu = ttk.Combobox(dropdown_frame, textvariable=interval_var, values=intervals)
interval_menu.grid(row=1, column=2)

ttk.Label(dropdown_frame, text="Select Financial Indicator:").grid(row=0, column=3)
indicator_menu = ttk.Combobox(dropdown_frame, textvariable=indicator_var, values=indicators)
indicator_menu.grid(row=1, column=3)

# Fetch Button
fetch_button = ttk.Button(root, text="Fetch Data", command=fetch_data_create_plot)
fetch_button.pack()

# Matplotlib figure
fig, ax = plt.subplots(figsize=(5,3))
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
canvas_frame = tk.Frame(root, width=700, height=500)
canvas_frame.pack(fill=tk.BOTH, expand=True)
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Start GUI
root.mainloop()
