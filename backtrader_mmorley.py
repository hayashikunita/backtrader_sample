import matplotlib.pylab as pylab  # グラフ描画用ライブラリ
import backtrader.analyzers as btanalyzers  # バックテストの解析用ライブラリ
import backtrader.feeds as btfeed  # データ変換
import backtrader as bt  # Backtrader
import calendar
import csv
import datetime
import os

import pandas as pd

"""  """
import numpy as np
from sqlalchemy import null

input_csv = os.path.join(
    os.getcwd(),
    "",
)

df = pd.read_csv(input_csv)


df = df.set_index(pd.to_datetime(df["DateTime"])).drop("DateTime", axis=1)


data = btfeed.PandasData(
    dataname=df,  # PandasのデータをBacktraderの形式に変換する
    fromdate=datetime.datetime(2011, 1, 1),  # 期間指定
    # todate=datetime.datetime(2020, 12, 31),# 期間指定
)


class ruin_fixed_amonunt:
    def __init__(self, win_pct, risk_reward, risk_rate):
        self.win_pct = win_pct
        self.risk_reward = risk_reward
        self.risk_rate = risk_rate
        if self.is_error():
            raise

    def is_error(self):
        if self.win_pct == 0 or self.risk_reward == 0 or self.risk_rate == 0:
            return True
        elif (
            not 0 <= self.win_pct <= 1
            or not 0 < self.risk_reward
            or not 0 <= self.risk_rate <= 1
        ):
            return True
        else:
            return False

    def equation(self, x, P, R):
        return P * x ** (R + 1) + (1 - P) - x

    def solve_equation(self):
        S, P, R = 0, self.win_pct, self.risk_reward
        while self.equation(S, P, R) > 0:
            S += 1e-4
        if S >= 1:
            S = 1
        return S

    def calc(self):
        S = self.solve_equation()
        return S ** (1 / self.risk_rate)


class ruin_fixed_rate:
    def __init__(self, win_pct, risk_reward, risk_rate, funds, ruin_line):
        self.win_pct = win_pct
        self.risk_reward = risk_reward
        self.risk_rate = risk_rate
        self.funds = funds
        self.ruin_line = ruin_line
        if self.is_error():
            raise

    def is_error(self):
        if (
            self.win_pct == 0
            or self.risk_reward == 0
            or self.risk_rate == 0
            or self.ruin_line == 0
        ):
            return True
        elif (
            not 0 <= self.win_pct <= 1
            or not 0 < self.risk_reward
            or not 0 <= self.risk_rate <= 1
            or self.funds < 0
            or self.ruin_line < 0
            or self.ruin_line > self.funds
        ):
            return True
        else:
            return False

    def equation(self, x, P, R):
        return P * x ** (R + 1) + (1 - P) - x

    def solve_equation(self, win_pct, R):
        S, P = 0, win_pct
        while self.equation(S, P, R) > 0:
            S += 1e-4
        if S >= 1:
            S = 1
        return S

    def calc(self):
        a = np.log(1 + self.risk_reward * self.risk_rate)
        b = abs(np.log(1 - self.risk_rate))
        n = np.log(self.funds / self.ruin_line)
        R = a / b
        S = self.solve_equation(self.win_pct, R)
        return S ** (n / b)


class myStrategy(bt.Strategy):  # ストラテジー

    params = dict(
        when=bt.timer.SESSION_START,
        timer=True,
        cheat=False,
        offset=datetime.timedelta(),
        repeat=datetime.timedelta(),
        weekdays=[],
    )
    n1 = 5  
    n2 = 10
    n3 = 500

    def log(self, txt, dt=None, doprint=False):  
        if doprint:
            print(
                "{0:%Y-%m-%d %H:%M:%S}, {1}".format(
                    dt or self.datas[0].datetime.datetime(0), txt
                )
            )


    def __init__(self): 

        self.sma1 = bt.indicators.SMA(
            self.data.close, period=self.n1
        )  
        self.sma2 = bt.indicators.SMA(
            self.data.close, period=self.n2
        )  
        self.sma3 = bt.indicators.SMA(
            self.data.close, period=self.n3
        )  # 

        self.tradehistory = []
        self.endingvaluelist = []

        # self.time_now = data[0].datetime.datetime(0)
        # self.time_now_month = self.time_now.month
        # self.time_now_day = self.time_now.day
        # self.time_now_hour = self.time_now.hour
        # self.time_now_minute = self.time_now.minute

        if self.p.timer:
            self.add_timer(
                when=self.p.when,
                offset=self.p.offset,
                repeat=self.p.repeat,
                weekdays=self.p.weekdays,
            )

        if self.p.cheat:
            self.add_timer(
                when=self.p.when,
                offset=self.p.offset,
                repeat=self.p.repeat,
                cheat=True,
            )

    def next(self):  

        _, isoweek, isoweekday = self.datetime.date().isocalendar()
        d_month = self.datetime.date().month
        d_day = self.datetime.date().day
        txt = "{},{},{}, Week {}, Day {}, O {}, H {}, L {}, C {}, position {}".format(
            d_month,
            d_day,
            self.datetime.datetime(),
            isoweek,
            isoweekday,
            self.data.open[0],
            self.data.high[0],
            self.data.low[0],
            self.data.close[0],
            self.position.size,
        )
        print(txt)

        self.lista = []
        self.listb = []

        for i in range(0, -7 - 1, -1):
            self.lista.append(self.data.high[i])
        self.highest = max(self.lista)

        for i in range(0, -7 - 1, -1):
            self.listb.append(self.data.low[i])
        self.lowest = min(self.listb)


        entrycondition1 = self.sma1 > self.sma2

        entrycondition2 = self.sma1 < self.sma2

        if (
            (
                self.data.datetime.time().hour == 9
                and self.data.datetime.time().minute == 00
            )
        ):
            if entrycondition1:
                if not self.position:  # ポジションを持っていない場合
                    self.buy_bracket(
                        limitprice=self.data.close[0] + 100,
                        price=self.data.close[0],
                        stopprice=self.data.close[0] - 100,
                        tradeid=1,
                        valid=datetime.timedelta(hours=3, minutes=00),
                    )
        if (
            (
                self.data.datetime.time().hour == 9
                and self.data.datetime.time().minute == 00
            )
        ):
            if entrycondition2:
                if not self.position:  # ポジションを持っていない場合
                    self.sell_bracket(
                        limitprice=self.data.close[0] - 250,
                        price=self.data.close[0],
                        stopprice=self.data.close[0] + 75,
                        tradeid=1,
                        valid=datetime.timedelta(hours=3, minutes=00),
                    )

        if (self.data.datetime.time().hour == 12) and (
            self.data.datetime.time().minute == 29
        ):
            if self.position:
                self.close(tradeid=1)  # ポジションをクローズする

    def stop(self):
        self.log(
            "(MA Period %2d) Ending Value %.2f" % (
                self.n1, self.broker.getvalue()),
            doprint=True,
        )


    def notify_order(self, order):  # 注文のステータスの変更を通知する
        if order.status in [order.Submitted, order.Accepted]:  # 注文の状態が送信済or受理済の場合
            return  # 何もしない

        if order.status in [order.Completed]:  # 注文の状態が完了済の場合
            if order.isbuy():  # 買い注文の場合
                self.log(
                    "買い約定, 取引量:{0:.2f}, 価格:{1:.2f}, 取引額:{2:.2f}, 手数料:{3:.2f}".format(
                        order.executed.size,  # 取引量
                        order.executed.price,  # 価格
                        order.executed.value,  # 取引額
                        order.executed.comm,  # 手数料
                    ),
                    dt=bt.num2date(order.executed.dt),  # 約定の日時をdatetime型に変換
                    doprint=True,  # Trueの場合出力
                )

            elif order.issell():  # 売り注文の場合
                self.log(
                    "売り約定, 取引量:{0:.2f}, 価格:{1:.2f}, 取引額:{2:.2f}, 手数料:{3:.2f}".format(
                        order.executed.size,  # 取引量
                        order.executed.price,  # 価格
                        order.executed.value,  # 取引額
                        order.executed.comm,  # 手数料
                    ),
                    dt=bt.num2date(order.executed.dt),  # 約定の日時をdatetime型に変換
                    doprint=True,  # Trueの場合ログを出力する
                )

            # 注文の状態がキャンセル済・マージンコール（証拠金不足）・拒否済の場合
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log("注文 キャンセル・マージンコール（証拠金不足）・拒否", doprint=True)

    def notify_trade(self, trade):  # 取引の開始/更新/終了を通知する
        if trade.isclosed:  # トレードが完了した場合
            self.log(
                "取引損益, 総額:{0:.2f}, 純額:{1:.2f}".format(
                    trade.pnl, trade.pnlcomm  # 損益  # 手数料を差し引いた損益
                ),
                doprint=True,  # Trueの場合ログを出力する
            )
            self.tradehistory.append(trade.pnl)

            # 累積利益分析
            getvalue = self.broker.get_value()
            tv_ruisekirieki = getvalue - startcash
            self.endingvaluelist.append(tv_ruisekirieki)
            # 累積利益分析

# バックテストの設定
cerebro = bt.Cerebro()  # Cerebroエンジンをインスタンス化


cerebro.addstrategy(myStrategy)  # ストラテジーを追加


# 時間足変更
# 時間足変更


cerebro.adddata(data)  # データをCerebroエンジンに追加

cerebro.addobserver(bt.observers.DrawDown)
# cerebro.addobserver(bt.observers.TimeReturn)
cerebro.addobserver(bt.observers.Value)
# cerebro.addobserver(bt.observers.FundValue)


cerebro.broker.setcash(200000)  # 所持金を設定

# cerebro.broker.setcommission(commission=0.000) # 手数料（スプレッド）を0.05%に設定

cerebro.addsizer(
    bt.sizers.SizerFix, stake=1
)  # デフォルト（buy/sellで取引量を設定していない時）の取引量を所持金に対する割合で指定する

startcash = cerebro.broker.getvalue()  # 開始時の所持金
cerebro.broker.set_coc(True)  # 発注時の終値で約定する

# 解析の設定

cerebro.addanalyzer(btanalyzers.DrawDown, _name="myDrawDown")  # ドローダウン
cerebro.addanalyzer(btanalyzers.SQN, _name="mySQN")  # SQN
cerebro.addanalyzer(btanalyzers.TradeAnalyzer,
                    _name="myTradeAnalyzer")  # トレードの勝敗等の結果
cerebro.addanalyzer(btanalyzers.AnnualReturn, _name="myAnnualReturn")
cerebro.addanalyzer(btanalyzers.SharpeRatio_A, _name="mySharpRatio")
cerebro.addanalyzer(btanalyzers.PeriodStats, _name="myPeriodStats")
cerebro.addanalyzer(btanalyzers.TimeDrawDown, _name="myTimeDrawDown")
cerebro.addanalyzer(
    btanalyzers.TimeDrawDown, timeframe=bt.TimeFrame.Months, _name="myMonthDrawDown"
)

cerebro.addanalyzer(btanalyzers.Calmar, _name="myCalmar")
cerebro.addanalyzer(btanalyzers.PositionsValue, _name="myPositionsValue")
cerebro.addanalyzer(btanalyzers.GrossLeverage, _name="myGrossLeverage")
cerebro.addanalyzer(btanalyzers.LogReturnsRolling, _name="myLogReturnsRolling")
cerebro.addanalyzer(btanalyzers.Returns, _name="myReturns")
cerebro.addanalyzer(
    btanalyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name="myTimeReturn"
)
cerebro.addanalyzer(btanalyzers.PyFolio, _name="myPyFolio")
cerebro.addanalyzer(btanalyzers.Transactions, _name="myTransactions")
cerebro.addanalyzer(btanalyzers.VWR, _name="myVWR")

cerebro.addobserver(bt.observers.Broker)

thestrats = cerebro.run()  # バックテストを実行
thestrat = thestrats[0]  # 解析結果の取得

print(thestrat)


# # 評価値の表示
# print('Start                 :{0:%Y/%m/%d %H:%M:%S}'.format( # ヒストリカルデータの開始日時
#     pd.to_datetime(df.index.values[0])
# ))
# print('End                   :{0:%Y/%m/%d %H:%M:%S}'.format (# ヒストリカルデータの開始日時
#     pd.to_datetime(df.index.values[-1])
# ))
# print('Duration              :{0}'.format( # ヒストリカルデータの期間の長さ
#     pd.to_datetime(df.index.values[-1]) - pd.to_datetime(df.index.values[0])
# ))

# print('Equity Final[$]       :{0:.2f}'.format( # 所持金の最終値（closedした取引）
#     startcash + thestrat.analyzers.myTradeAnalyzer.get_analysis().pnl.net.total
# ))
# print('Return[%]             :{0:.2f}'.format( # 利益率=損益÷開始時所持金×100
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().pnl.net.total / startcash * 100
# ))

# print('Max. Drawdown[%]      :{0:.2f}'.format( # 最大下落率
#     thestrat.analyzers.myDrawDown.get_analysis().max.drawdown
# ))
# print('Max. Drawdown Duration:{0}'.format( # 最大下落期間
#     pd.Timedelta(minutes=thestrat.analyzers.myDrawDown.get_analysis().max.len)
# ))
# print('Trades                :{0}'.format( # 取引回数
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().total.closed
# ))
# winrate = ( # 勝率
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().won.total
#     / thestrat.analyzers.myTradeAnalyzer.get_analysis().total.closed
# )
# lostrate = ( # 敗率
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().lost.total
#     / thestrat.analyzers.myTradeAnalyzer.get_analysis().total.closed
# )
# print('Win Rate[%]           :{0:.2f}'.format( # 勝率=勝ち取引回数÷全取引回数×100
#     winrate * 100
# ))
# print('Best Trade[%]         :{0:.2f}'.format( # 1回の取引での利益の最大値÷所持金×100
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().won.pnl.max / startcash * 100
# ))
# print('Worst Trade[%]        :{0:.2f}'.format( # 1回の取引での損失の最大値÷所持金×100
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().lost.pnl.max / startcash * 100
# ))
# print('Avg. Trade[%]         :{0:.2f}'.format( # 損益の平均値÷所持金×100
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().pnl.net.average / startcash * 100
# ))
# print('Max. Trade Duration   :{0}'.format( # 1回の取引での最長期間
#     pd.Timedelta(minutes=thestrat.analyzers.myTradeAnalyzer.get_analysis().len.max)
# ))
# print('Avg. Trade Duration   :{0}'.format( # 1回の取引での平均期間
#     pd.Timedelta(minutes=thestrat.analyzers.myTradeAnalyzer.get_analysis().len.average)
# ))
# print('Expectancy[%]         :{0:.2f}'.format( # 期待値=平均利益×勝率＋平均損失×敗率
#     thestrat.analyzers.myTradeAnalyzer.get_analysis().won.pnl.average * winrate
#     + thestrat.analyzers.myTradeAnalyzer.get_analysis().lost.pnl.average * lostrate
# ))
# print('SQN                   :{0:.2f}'.format( # SQN システムの評価値
#     thestrat.analyzers.mySQN.get_analysis().sqn
# ))

# グラフの設定
# matplotlib inline
# ↑グラフをNotebook内に描画する


def convert_to_protra(thestrat, df):
    ana = thestrat.analyzers.myTradeAnalyzer.get_analysis()
    sqn = thestrat.analyzers.mySQN.get_analysis()
    dd = thestrat.analyzers.myDrawDown.get_analysis()
    timedrawdown = thestrat.analyzers.myTimeDrawDown.get_analysis()
    monthdrawdown = thestrat.analyzers.myMonthDrawDown.get_analysis()
    anr = thestrat.analyzers.myAnnualReturn.get_analysis()
    sharp = thestrat.analyzers.mySharpRatio.get_analysis()
    anana = thestrat.analyzers.myPeriodStats.get_analysis()

    # 未使用
    calmar = thestrat.analyzers.myCalmar.get_analysis()
    positionsvalue = thestrat.analyzers.myPositionsValue.get_analysis()
    GrossLeverage = thestrat.analyzers.myGrossLeverage.get_analysis()
    logreturnsrolling = thestrat.analyzers.myLogReturnsRolling.get_analysis()
    returns = thestrat.analyzers.myReturns.get_analysis()
    timereturn = thestrat.analyzers.myTimeReturn.get_analysis()
    # pyfolio = thestrat.analyzers.myPyFolio.get_analysis()
    # pyfolioshow = thestrat.analyzers.myPyFolio.get_pf_items()
    # pypolio
    transactions = thestrat.analyzers.myTransactions.get_analysis()
    vwr = thestrat.analyzers.myVWR.get_analysis()
    # 未使用

    # Pyfolio Integration

    # returns, positions, transactions, gross_lev = thestrat.analyzers.myPyFolio.get_pf_items()
    # print('-- RETURNS')
    # print(returns)
    # print('-- POSITIONS')
    # print(positions)
    # print('-- TRANSACTIONS')
    # print(transactions)
    # print('-- GROSS LEVERAGE')
    # print(gross_lev)

    # import pyfolio as pf
    # pf.create_full_tear_sheet(
    #     returns,
    #     positions=positions,
    #     transactions=transactions,
    #     # gross_lev=gross_lev,
    #     live_start_date='2011-01-01',
    #     round_trips=True
    # )

    # Pyfolio Integration

    # periodreturn =  thestrat.analyzers.PeriodStats.get_analysis()
    # Python3.6以降のfプリフィクスを用いた書き方（簡潔・低負荷）
    print()
    print(
        f"{pd.to_datetime(df.index.values[0]):%Y/%m/%d}～{pd.to_datetime(df.index.values[-1]):%Y/%m/%d}"
    )
    print("----------------------------------------")
    print(f"シンボル               NIKKEI225mini")
    print(f"条件                 {startcash}でmini1枚とする")
    print(f"初期資金              {startcash}円")
    print(
        f"最終資金              {startcash + thestrat.analyzers.myTradeAnalyzer.get_analysis().pnl.net.total}円"
    )
    print(
        f"税引後最終資金         {startcash + ((thestrat.analyzers.myTradeAnalyzer.get_analysis().pnl.net.total)*0.8)}円"
    )
    print("----------------------------------------")
    print(f"全トレード数           {ana.total.closed:}")
    print(
        f"勝ちトレード数(勝率)    {ana.won.total}({ana.won.total / ana.total.closed * 100:.2f}%)"
    )
    print(
        f"負けトレード数(負率)    {ana.lost.total}({ana.lost.total / ana.total.closed * 100:.2f}%)"
    )
    print()
    print(f"勝ちトレード最大利率    {ana.won.pnl.max/startcash:,.0f}")
    print(f"負けトレード最大損率    {ana.lost.pnl.max/startcash:,.0f}")
    print()
    print(f"全トレード平均利益      {ana.pnl.net.average:,.0f}円")
    print(f"勝ちトレード平均利益    {ana.won.pnl.average:,.0f}円")
    print(f"負けトレード平均損失    {ana.lost.pnl.average:,.0f}円")
    print()
    print(f"勝ちトレード最大利益    {ana.won.pnl.max:,.0f}円")
    print(f"負けトレード最大損益    {ana.lost.pnl.max:,.0f}円")
    print()
    print(f"全トレード平均期間      {ana.len.average:.2f}")
    print(f"勝ちトレード平均期間    {ana.len.won.average:.2f}")
    print(f"負けトレード平均期間    {ana.len.lost.average:.2f}")
    print(f"純利益                {ana.pnl.net.total:,.0f}円")
    print(f"勝ちトレード総利益      {ana.won.pnl.total:,.0f}円")
    print(f"負けトレード総損失      {ana.lost.pnl.total:,.0f}円")
    print("----------------------------------------")
    print(f"必要資金              {startcash:,.0f}円")
    print(f"最大ポジション         1")
    print("----------------------------------------")
    print(f"プロフィットファクター   {abs(ana.won.pnl.total / ana.lost.pnl.total):.2f}")
    print(
        f"バイアンドホールド      {0:.2f}".format(
            (df["Close"][-1] - df["Close"][0]) / df["Close"][0] * 100
        )
    )
    print(f"全期間利益率（単利）     {ana.pnl.net.total / startcash * 1000:,.2f}%")
    print(
        f"エクスポージャー        {0:.2f}".format(
            thestrat.analyzers.myTradeAnalyzer.get_analysis().len.total / len(df) * 100
        )
    )  # ポジションを持っていた期間の割合（ポジションを持っていた期間÷全期間×100）
    print("----------------------------------------")
    print(f"最大ドローダウン(簿価)   {dd.moneydown:,.0f}円")
    print(f"最大ドローダウン(時価)   {dd.max.moneydown:,.0f}円")
    print(f'最大ドローダウン期間     {timedrawdown["maxdrawdownperiod"]:,.0f}日')
    print("----------------------------------------")
    print(f"現在進行中のトレード数   {ana.total.open}")
    print("----------------------------------------")
    print(f"SQN(システムの評価値)   {sqn.sqn:.2f}")
    print(f'シャープレシオ          { sharp["sharperatio"]:.2f}')
    print(
        f"リスクリワードレシオ     {(ana.won.pnl.average/((-1)*ana.lost.pnl.average)):,.2f}")
    #   資金率（損失の許容額 ÷ トレード総資産 × 100）
    # 「1回の取引で失う最大損失と総資金の比率」
    print(f"資金率                 {(-1)*ana.lost.pnl.max/startcash*10000:,.2f}%")

    annualreturn = anana["average"] * 1000
    annualstddev = anana["stddev"] * 1000

    print(f"年間平均利率(%)          {annualreturn:.2f}")
    print(f"年間平均標準偏差(%)       {annualstddev:.2f} ")
    # 破産確率

    # バルサラ破産確率
    # バルサラ破産確率

    win_pct = (
        thestrat.analyzers.myTradeAnalyzer.get_analysis().won.total
        / thestrat.analyzers.myTradeAnalyzer.get_analysis().total.closed
    )  # 勝率
    risk_reward = ana.won.pnl.average / \
        ((-1) * ana.lost.pnl.average)  # リスクリワード比率
    risk_rate = (ana.lost.pnl.max * (-1)) / startcash * 100  # 1回のトレードで取るリスク率
    funds = startcash  # 初期資金
    ruin_line = startcash * 0.3  # 撤退ライン（破産）
    print(win_pct)
    print(risk_reward)
    print(risk_rate)

    ruin_ratea = ruin_fixed_amonunt(win_pct, risk_reward, risk_rate).calc()
    print(f"破産確率(定額)          { ruin_ratea :.2%}")

    print(ruin_ratea)
    ruin_rateb = ruin_fixed_rate(
        win_pct, risk_reward, risk_rate, funds, ruin_line
    ).calc()
    print(f"破産確率（定率）         {ruin_rateb :.2%}")
    print(ruin_rateb)
    # バルサラ破産確率


    # 年度別、勝率、最大DD、  年度	取引回数	運用損益	年利	勝率	PF	 最大DD

    print("----------------------------------------")
    # print(f'anualreturn   {anr:.2f}')
    print(f"[年度別レポート] ")

    for i in range(2011, 2021, 1):
        if i == 2011:
            for j in range(8, 13, 1):
                k = int(calendar.monthrange(i, j)[1])
                print(
                    f"{i:<2}年度{j:<2}月 年間運用損益 {anr[i] * startcash*100:,.0f}円 年利回り(単利) {((anr[i] * startcash*100) /startcash)*100:,.2f}% 月間運用損益 {(timereturn[datetime.datetime(i, j ,k,0,0)])*startcash*100:,.0f}円 月利回り（単利) {timereturn[datetime.datetime(i, j ,k,0,0)]*10000:,.2f}%"
                )
        else:
            for j in range(1, 13, 1):
                k = int(calendar.monthrange(i, j)[1])
                print(
                    f"{i:<2}年度{j:<2}月 年間運用損益 {anr[i] * startcash*100:,.0f}円 年利回り(単利) {((anr[i] * startcash*100) /startcash)*100:,.2f}% 月間運用損益 {(timereturn[datetime.datetime(i, j ,k,0,0)])*startcash*100:,.0f}円 月利回り(単利) {timereturn[datetime.datetime(i, j ,k,0,0)]*10000:,.2f}%"
                )
    # 未使用
    # print("----------------------------------------")
    # print(anana)
    # print("----------------------------------------")
    # print(calmar)
    # print("----------------------------------------")
    # print(timedrawdown)
    # print("----------------------------------------")
    # print(monthdrawdown)
    # print("----------------------------------------")
    # print(positionsvalue)
    # print("----------------------------------------")
    # print(GrossLeverage)
    # print("----------------------------------------")
    # print(logreturnsrolling)
    # print("----------------------------------------")
    # print(returns)
    # print("----------------------------------------")
    # print(timereturn)
    # print("----------------------------------------")
    # print(pyfolio)
    # print("----------------------------------------")
    # print(transactions)
    # print("----------------------------------------")
    # print(vwr)
    # print("----------------------------------------")
    # 未使用

    # 累積利益分析
    print(thestrat.endingvaluelist)
    stock = thestrat.endingvaluelist

    arr = np.array(stock)
    matrix = arr.reshape((1, len(arr)))
    stock_row = matrix.transpose()

    with open("stock_row.csv", "w", encoding="Shift_jis") as f:  
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(stock_row)  # c
    # 累積利益分析
    # print(f'stddev   {periodreturn.stddev:.2f}')
    # pprint.pprint(thestrat.analyzers.myTradeAnalyzer.get_analysis())
    # pprint.pprint(thestrat.analyzers.myAnnualReturn.get_analysis()


convert_to_protra(thestrat, df)


pylab.rcParams["figure.figsize"] = 12, 8  # グラフのサイズ
cerebro.plot(
    style="candle",  # ロウソク表示にする
    barup="green",
    barupfill=False,  # 陽線の色、塗りつぶし設定
    bardown="red",
    bardownfill=False,  # 陰線の色、塗りつぶし設定
    fmt_x_data="%Y-%m-%d %H:%M:%S",  # 時間軸のマウスオーバー時の表示フォーマット
)

