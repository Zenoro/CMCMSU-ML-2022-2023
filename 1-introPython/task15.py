from typing import List


def hello(name=None) -> str:
    if not name:
        return 'Hello!'
    else:
        return "Hello, " + name + "!"


def int_to_roman(num: int) -> str:
    romdict={
        1:"I",
        5:"V",
        10:"X",
        50:"L",
        100:"C",
        500:"D",
        1000:"M"
    }
    
    #number of tens
    t_count = 1
    
    while num//10 >= t_count:
        t_count *= 10
    
    res = ""
    
    while num:
        dquant = int (num/t_count)    #quantity of digit
        if dquant <= 3:
            res += (romdict[t_count]*dquant)
        elif dquant == 4:
            res += (romdict[t_count]+romdict[t_count*5])
        elif dquant >= 5 and dquant <= 8:
            res += (romdict[t_count*5] + (romdict[t_count] * (dquant-5)))
        elif dquant == 9:
            res += (romdict[t_count]+romdict[t_count*10])

        num = num % t_count
        t_count /= 10

    return res


def longest_common_prefix(strs_input: List[str]) -> str:
    res = ""

    if strs_input != []:
        for j in range (0, len(strs_input)):
            strs_input[j] = strs_input[j].strip()
      
        strs_input.sort(key=len)
      
        count = 0
        for i in range (0,len(strs_input[0])):
            buf = strs_input[0][i]
            for j in range (0, len(strs_input)):
                if buf == strs_input[j][i]:
                    count += 1
                else:
                    count = 0
            if count == len(strs_input):
                count = 0
                res += buf
        return res
    
    else:
        return ""


def primes() -> int:
    n = 100000
    a = list(range(n+1))
    a[1] = 0
    lst = []

    i = 2
    while i <= n:
        if a[i] != 0:
            yield a[i]
            for j in range(i, n+1, i):
                a[j] = 0
        i += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = -1):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    
    def __repr__(self):
        return f'To learn the balance call balance.'

    def __call__(self, sum_spent: int):
        if sum_spent > self.total_sum:
            print("Can't spend "+ str(sum_spent) +" dollars")
            raise ValueError 
        else:
            self.total_sum -= sum_spent
            print("You spent " + str(sum_spent) + " dollars")

    def __add__(self, other):
        added = BankCard(0, 0)
        if isinstance (other, BankCard):
            added.total_sum = self.total_sum+other.total_sum
            if self.balance_limit>=0 and other.balance_limit>=0:
                added.balance_limit = max(self.balance_limit, other.balance_limit)
            else:
                added.balance_limit = -1
        return added
    
    def put(self, sum_put: int):
        self.total_sum += sum_put
        print("You put " + str(sum_put) + " dollars.")

    @property
    def balance(self):
        if self.balance_limit < 0:
            return (self.total_sum)
        else:
            if self.balance_limit > 0:
                self.balance_limit -= 1
                return (self.total_sum)
            else: 
                print("Balance check limits exceeded.")
                raise ValueError

