import datetime
from dateutil.relativedelta import relativedelta
import random
import copy
import collections
import numpy as np
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

LIBOR = .025

#helper function for irr, calculates net present value given cash flows and an interest rate
def net_present_value(ir, cfs):
    yrs = [x/4 for x in range(len(cfs))]
    yrs = np.asarray(yrs)
    return np.sum(cfs / (1. + ir) ** yrs)

#finds the internal rate of return of a CLO or fund
def irr(cfs, **kwargs):
    irr = fsolve(net_present_value, x0=0.03, args=(cfs), maxfev = 10000, **kwargs, full_output = True)
    if abs(irr[1]['fvec']) < 1:
        return np.asscalar(irr[0])
    else:
        return -.5

#general method for creating 2d plots of IRR vs default probability
#list, list, string
def plot2d(x, y, title, fname):
    plt.plot(x, y, 'x', color = 'green')
    plt.xlabel('Annualized Default Probability')
    plt.ylabel('IRR')
    plt.ylim(-.5,.3)
    plt.title(title)
    plt.savefig(fname)
    plt.clf()

#takes a model as a paramter, creates a 3d scatter plot
def scatter3d(r, title, fname):
    x, y, z = [], [], []
    x1, y1, z1 = [], [], []
    for a, b, c in r:
        for q in c:
            if q is not None:
                if q >= 0:
                    x.append(a)
                    y.append(b)
                    z.append(q)
                else:
                    x1.append(a)
                    y1.append(b)
                    z1.append(q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ax.scatter(x, y, z, c = 'g', marker = 'o')
    ax.scatter(x1, y1, z1, c = 'r', marker = 'o')

    ax.set_title(title)
    ax.set_xlabel('Default Probability')
    ax.set_ylabel('Recovery Rate')
    ax.set_zlabel('IRR')
    plt.savefig(fname)
    plt.show()

#makes the dependent variable (irr) for modeling purposes
def dependent(recovery_rate, n, share_price = None):
    return irr(Fund(recovery_rate, share_price).gen_cash_flows())

class Loan:

    #integer, float, datetime
    def __init__(self, size, margin, maturity, default_prob):
        self.size = size
        self.margin = margin
        self.maturity = maturity
        self.default_prob = default_prob
        self.defaulted = False

    #creates a list of dictionaries which represents a loan
    def gen_cash_flow(self, start_date, default_prob, recovery_rate):
        self.cash_flows = []
        default_prob = (1-(1-self.default_prob)**(1/4))
        while len(self.cash_flows)==0 or self.cash_flows[-1]['size']>0:
            self.cash_flows.append({})
            self.cash_flows[-1]['date'] = start_date
            start_date = start_date + relativedelta(months = 3)
            self.cash_flows[-1]['size'] = self.size
            if self.defaulted:
                self.cash_flows[-1]['interest'] = 0
                self.cash_flows[-1]['principal'] = 0
                self.cash_flows[-1]['writedown'] = self.size * (1-recovery_rate)
                self.cash_flows[-1]['recovery'] = self.size * recovery_rate
                self.cash_flows[-1]['size'] = 0
            else:
                interest = self.size * (self.margin + LIBOR) / 4
                self.cash_flows[-1]['interest'] = interest
                self.cash_flows[-1]['writedown'] = 0
                self.cash_flows[-1]['recovery'] = 0

                if start_date > self.maturity:
                    self.cash_flows[-1]['principal'] = self.size
                    self.cash_flows[-1]['size'] = 0
                else:
                    self.cash_flows[-1]['principal'] = 0

            if random.random() < default_prob:
                self.defaulted = True

    #basic print function for loans
    def __str__(self):
        s = 'Size            interest   principal   writedown    recovery \n'
        for cf in self.cash_flows:
            s += '{:12,.0f}'.format(cf['size'])
            s += '{:12,.0f}'.format(cf['interest'])
            s += '{:12,.0f}'.format(cf['principal'])
            s += '{:12,.0f}'.format(cf['writedown'])
            s += '{:12,.0f}'.format(cf['recovery'])
            s += '\n'
        return s

    #function to project forward cash flows for a list of loans
    def proj_cash_flows(loans, start_date, default_prob, recovery_rate):
        for loan in loans:
            loan.gen_cash_flow(start_date, default_prob, recovery_rate)

    #function to consolidate projected cash flows by adding them
    def consolidate(loans):
        result = []
        for loan in loans:
            for n, cash_flows in enumerate(loan.cash_flows):
                if len(result) <= n:
                    result.append({'size': 0, 'principal': 0, 'interest': 0, 'writedown': 0, 'recovery': 0})
                result[n]['size'] += cash_flows['size']
                result[n]['principal'] += cash_flows['principal']
                result[n]['interest'] += cash_flows['interest']
                result[n]['writedown'] += cash_flows['writedown']
                result[n]['recovery'] += cash_flows['recovery']
        return result

    #returns the irr of a loan, taking the loans cash flows as a parameter
    def loan_irr(cfs):
        result = [-cfs[0]['size']]
        for cf in cfs:
            result.append(cf['interest'] + cf['principal'] + cf['recovery'])
        return irr(result)

    #creates two lists of sample default probabilities and irrs that can be fed into the 2dplot function
    #integer, integer, float, float
    def sample_irrs(iterations, loan_count, default_prob_low, default_prob_high):
        x, y = [], []
        for n in range(iterations):
            ls = [Loan(5000000, random.uniform(.03, .045), datetime.date(2022, 1, 1) + relativedelta(months = random.randint(0, 24))) for x in range(loan_count)]
            default_prob = np.random.uniform(default_prob_low, default_prob_high)
            for l in ls:
                l.gen_cash_flow(datetime.date(2019,7,1), default_prob, .5)
            cf = Loan.consolidate(ls)
            x.append(default_prob)
            y.append(Loan.loan_irr(cf))
        return x, y

class CLO:

    #pre-input default prob and recovery rate are meant to be overridden, in there for simplicity
    #creates tranches with sizes based on an actual CLO
    def __init__(self, recovery_rate = .5, default_prob = {"tech": .1, "health": 0.1, "standard": 0.02}):
        self.default_prob = default_prob
        self.recovery_rate = recovery_rate
        self.assets = [Loan((1 + random.randint(2, 6)) * 1000000, random.uniform(.03, .045), datetime.date(2022, 1, 1) + relativedelta(months = random.randint(0, 24)), self.default_prob["standard"]) for x in range(100)]
        m = 0
        #scans through a dictionary of default probabilities, grabs loans, assigns them industry-specific default probs
        for key, value in self.default_prob.items():
            if key is not "standard":
                n = int(len(self.assets) * Fund.holdings[key])
                for x in range(n):
                    self.assets[m + x].default_prob = self.default_prob[key]
                m += n
        s = self.size()
        #tranches by rating, Q is equity
        self.tranches = collections.OrderedDict([('A', {'size': s * .62, 'margin': .015, 'OCtrigger': 1.216}),
                                                 ('B', {'size': s * .14, 'margin': .02,  'OCtrigger': 1.216}),
                                                 ('C', {'size': s * .07, 'margin': .0285,'OCtrigger': 1.135}),
                                                 ('D', {'size': s * .05, 'margin': .039, 'OCtrigger': 1.086}),
                                                 ('E', {'size': s * .04, 'margin': .0695,'OCtrigger': 1.047}),
                                                 ('Q', {'size': s * .08})])
        for tranche in self.tranches.values():
            tranche['initial_size'] = tranche['size']
        self.cash_flows = self.gen_cash_flows(datetime.date(2019,7,1))

    #returns total size of a CLO
    def size(self):
        total = 0
        for loan in self.assets:
            total += loan.size
        return total

    #returns a list that represents the cash flows of a CLO
    def gen_cash_flows(self, start_date = datetime.date(2019,7,1)):
        self.result = []
        Loan.proj_cash_flows(self.assets, start_date, self.default_prob, self.recovery_rate)
        self.periods = Loan.consolidate(self.assets)
        #period: consolidated loan cashflows in one time step
        for period in self.periods:
            total_tranche_balance = 0
            for tranche in self.tranches.values():
                if tranche['size'] > 0:
                    total_tranche_balance += tranche['size']
                    tranche['OC'] = period['size'] / total_tranche_balance
                    if 'OCtrigger' in tranche.keys():
                        tranche['triggered'] = tranche['OC'] < tranche['OCtrigger']
                        tranche['paydown'] = max(total_tranche_balance-period['size']/tranche['OCtrigger'],0)
                    else:
                        tranche['triggered'] = False
                else:
                    tranche['OC'] = 0
            loan_principal = period['principal'] + period['recovery']
            loan_interest = period['interest']
            loan_loss = period['writedown']
            for tranche in self.tranches.values():
                tranche['principal'] = 0
                if 'margin' in tranche.keys():
                    tranche['interest'] = (tranche['size'] * (LIBOR + tranche['margin'])) / 4
                    loan_interest -= tranche['interest']
                else:
                    for tranche in self.tranches.values():
                        # if OC trigger is breached, pay down tranche to get back into compliance
                        if 'triggered' in tranche.keys() and tranche['triggered']:
                            additional_prin = min(loan_interest,tranche['paydown'], tranche['size'])
                            tranche['principal'] += additional_prin
                            tranche['size'] -= additional_prin
                            loan_interest -= additional_prin
                        else:
                            tranche['paydown'] = 0
                    # class Q receives all remaining interest
                    tranche['interest'] = max(loan_interest, 0)
                prin_payment = min(loan_principal, tranche['size'])
                tranche['size'] -= prin_payment
                tranche['principal'] += prin_payment
                loan_principal -= prin_payment
            for tranche in reversed(self.tranches.values()):
                prin_loss = min(loan_loss, tranche['size'])
                tranche['size'] -= prin_loss
                tranche['writedown'] = prin_loss
                loan_loss -= prin_loss
            self.result.append(copy.deepcopy(self.tranches))

        return self.result

    #helper function for CLO print functions
    def get_cash_flows(self, key):
        result = []
        for cf in self.result:
            result.append((cf[key]['size'], cf[key]['interest'], cf[key]['OC'], cf[key]['triggered'], cf[key]['principal'], cf[key]['writedown'], cf[key]['paydown']))
        return result

    #gets cash flows for a specific tranche as a vector
    def get_net_cash_flows(self, key):
        size = []
        interest = []
        principal = []
        for cf in self.result:
            size.append(cf[key]['size'])
            interest.append(cf[key]['interest'])
            principal.append(cf[key]['principal'])
        return size, interest, principal

    #returns a dictionary of irrs for existing cash flows
    def irrs(self):
        return {k : self.clo_irr(k) for k in self.tranches.keys()}

    #helper function for irrs
    def clo_irr(self, k):
        result = [-self.cash_flows[0][k]['size']]
        for cf in self.cash_flows:
            result.append(cf[k]['interest'] + cf[k]['principal'])
        return self.default_prob, self.recovery_rate, irr(result)

    #returns monte carlo results for one or more iterations
    def get_mc_results(n, default_high=.1, default_low=0, recovery_high=.5, recovery_low=.5):
        r = []
        for i in range(n):
            default = np.random.uniform(default_low,default_high)
            recovery = np.random.uniform(recovery_low,recovery_high)
            r.append(CLO(default,recovery).irrs())
        return r

    #generates 2d plots by tranche for specified monte carlo run
    def plot_mc_results(n, fname, default_high=.1, default_low=0, recovery_high=.5, recovery_low=.5):
        m = CLO.get_mc_results(n, default_high, default_low, recovery_high, recovery_low)
        result = {}
        for irrs in m:
            for key in irrs:
                if key not in result.keys():
                    result[key] = ([],[])
                result[key][0].append(irrs[key][0])
                result[key][1].append(irrs[key][2])
        for key in result:
            x = [result[key][0]]
            y = [result[key][1]]
            plot2d(x,y,'Tranche: ' + key, fname)

    #print method for CLO
    def __str__(self):
        s = ''
        for i, period in enumerate(self.periods):
            s += str(i) + ' '
            s += str(period)
            s += '\n'
        for key in self.tranches.keys():
            s += 'Size,     interest,     OC,     triggered,    principal,     writedown,     paydown'
            s += '\n'
            r = self.get_cash_flows(key)
            for i, q in enumerate(r):
                s += str(i) + ' '
                s += key + ' ' + str(q)
                s += '\n'
        return s

class Fund:

    #% of ECC's holdings in each of ECC's top 10 industries (May 2019)
    #for example, 10.6% of ECC's CLOs are in the tech industry
    holdings = {}
    holdings["tech"] = .106
    holdings["health"] = .082
    holdings["publishing"] = .07
    holdings["telecomm"] = .054
    holdings["finance"] = .053
    holdings["hotel"] = .05
    holdings["commercial"] = .042
    holdings["development"] = .034
    holdings["utilities"] = .034
    holdings["chem"] = .032

    #Fund is modeled after ECC
    #all the numbers seen here are from ECCs financial statements
    #a share price of "None" sets the equity cap of ECC to their NAV
    def __init__(self, recovery_rate, share_price = None):
        self.assets = 551031808
        self.equity_positions = 74
        self.debt_positions = 17
        self.positions = self.equity_positions + self.debt_positions
        self.senior_liabilities = (67815896, 31625000, 47118150, 45450000)
        self.net_assets = self.assets - np.sum(self.senior_liabilities)
        self.clo_list = [CLO(recovery_rate) for x in range(self.positions)]
        self.position_size = self.assets / self.positions
        self.senior_coupons = (.066875, .0675, .0775, .0775)
        self.equity_cap = self.assets - np.sum(self.senior_liabilities) if share_price is None else share_price * 23647000

    #principal payments go to the most senior first
    #interest payments go to most senior left, pay amount of interest due and the work your way up
    def gen_cash_flows(self):
        #copy the balance of the amount of the size of the liabilites into a list
        liabilities = [x for x in self.senior_liabilities]
        result = [-self.equity_cap]
        balance, interest, principal = self.get_asset_cash_flows()
        for n in range(len(balance)):
            senior_interest = np.dot(liabilities, self.senior_coupons) / 4
            payment1 = min(senior_interest, interest[n])
            senior_interest -= payment1
            interest[n] -= payment1
            payment2 = min(senior_interest, principal[n])
            senior_interest -= payment2
            principal[n] -= payment2
            #use available principal to pay down liabilities
            for m in range(len(liabilities)):
                payment3 = min(liabilities[m], principal[n])
                liabilities[m] -= payment3
                principal[n] -= payment3
            result.append(interest[n] + principal[n])
        return result


    #consolidates cash flows in a Fund
    def get_asset_cash_flows(self):
        max_length = 0
        for n, clo in enumerate(self.clo_list):
            if n < self.equity_positions:
                max_length = max(max_length, len(clo.get_net_cash_flows('Q')[0]))
            else:
                max_length = max(max_length, len(clo.get_net_cash_flows('E')[0]))
        size = [0 for x in range(max_length)]
        interest = [0 for x in range(max_length)]
        principal = [0 for x in range(max_length)]
        for n, clo in enumerate(self.clo_list):
            if n < self.equity_positions:
                s, i, p = clo.get_net_cash_flows('Q')
                scale = self.position_size / clo.tranches['Q']['initial_size']
            else:
                s, i, p = clo.get_net_cash_flows('E')
                scale = self.position_size / clo.tranches['E']['initial_size']
            for n in range(len(s)):
                size[n] += s[n] * scale
                interest[n] += i[n] * scale
                principal[n] += p[n] * scale
        return size, interest, principal

    #creates a model with random default prob and recovery rate rather than uniform
    def random_model3d(iterations, share_price = None):
        result = []
        for x in range(iterations):
            print(x,iterations)
            d = np.random.uniform(0,.06)
            r = np.random.uniform(.4,.7)
            result.append((d, r, dependent(d, r, 1, share_price)))
        return result

    #linspace, linspace, integer, float
    #creates a list that contains the probability of defaulting, the recovery rate, and the subsequent irr
    def model3d(default_probs, recovery_rates, n = 5, share_price = None):
        result = []
        for default_prob in default_probs:
            for recovery_rate in recovery_rates:
                print(default_prob, recovery_rate)
                result.append((default_prob, recovery_rate, dependent(default_prob, recovery_rate, n, share_price)))
        return result

    #models a fund in 2d with a random default probability
    def random_model2d(recovery_rate, iterations, share_price = None):
        result = []
        for x in range(iterations):
            print(x+1, iterations)
            result.append(dependent(recovery_rate, 1, share_price))
        return result

    #x-axis is default probability, y-axis is irr, recovery rate is static
    #draws a 2d picture for a fund
    def scatter2d(r, title, fname):
        x, y = [], []
        for a, b in r:
            for q in b:
                if q is not None:
                    x.append(a)
                    y.append(q)

        plot2d(x,y, title, fname)

    #linspace, linspace
    #creates a table
    def tabulator(default_probs, recovery_rates, n = 30, share_price = None):
        for d in default_probs:
            for r in recovery_rates:
                irrs = dependent(d, r, n, share_price)
                print(d, r, np.average(irrs), np.std(irrs))


'''
ECC capital structure in order of seniority:
series 2028 notes
series 2027 notes
preferred stock A
preferred stock B
common stock
'''
