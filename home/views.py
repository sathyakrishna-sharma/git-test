from django.shortcuts import HttpResponse

import openai
import math
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import random
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
#from .assets.calculations import *



import math

def sin(x):
    return math.sin(math.radians(x))


def cos(x):
    return math.cos(math.radians(x))


def tan(x):
    return math.tan(math.radians(x))


def ln(x):
    return math.log2(math.radians(x))


def log(x):
    return math.log10(x)


def fact(x):
    return math.factorial(x)


def root(x):
    return math.sqrt(x)


def pow(x, y):
    return math.pow(x, y)



# Create your views here.
def index(request):
    return HttpResponse("this is home-page")



@csrf_exempt
def combined_api(request):
    # Initializing global variables
    evaluate_math = ""
    category = ""
    real_life_examples = ""
    level_1_questions = ""
    level_2_questions = ""
    level_3_questions = ""
    specific_category_questions = ""

    openai.api_key = ' '

    prompt_template = '''
    Let's solve mathematical word problems in a careful, formal manner. The solution will follow the Peano format: 
    1- Each sentence in the solution either introduces a new variable or states a new equation. 
    2- The last sentence gives the goal: which variable will contain the answer to the problem. 
    3- Each equation only uses previously introduced variables. 
    4- Each quantity is only named by one variable.
    5- Use all the numbers in the question.
    Q: Mary had 5 apples. The next morning, she ate 2 apples. Then, in the afternoon, she bought as many apples as she had after eating those apples in the morning. How many apples did she end up with?
    Peano solution:
    Let a be the number of apples Mary started with [[var a]]. We have [[eq a = 5]]. 
    Let b be how many apples she had in the morning after eating 2 apples [[var b]]. We have [[eq b = a - 2]]. 
    Let c be the apples she bought in the afternoon [[var c]]. 
    Since she bought as many as she had after eating, we have [[eq c = b]]. 
    Let d be how many apples she ended up with [[var d]]. We have [[eq d = b + c]]. 
    The answer is the value of d [[answer d]]. 
    Q: Mario and Luigi together had 10 years of experience in soccer. Luigi had 3 more than Mario. How many did Mario have?
    Peano solution:
    Let a be the number of years Mario had [[var a]]. 
    Let b be the number of years Luigi had [[var b]]. We have [[eq a + b = 10]]. We also have [[eq b = a + 3]]. 
    The answer is the value of a [[answer a]].
    Q: The planet Goob completes one revolution after every 2 weeks. How many hours will it take for it to complete half a revolution?
    Peano solution:
    Let a be the number of hours in a week [[var a]]. We have [[eq a = 168]]. 
    Let b be the number of hours in a revolution [[var b]]. We have [[eq b = a * 2]]. 
    Let c be the number of hours in half a revolution [[var c]]. We have [[eq c = b / 2]]. 
    The answer is the value of c [[answer c]].
    Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
    Peano solution:
    Let a be the number of cars in the parking lot [[var a]]. We're given [[eq a = 3]]. 
    Let b be the number of cars arrived [[var b]]. We're given [[eq b = 2]]. 
    Let c be the number of cars in the parking lot now [[var c]]. We have [[eq c = a + b]]. 
    The answer is the value of c [[answer c]].
    Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    Peano solution:
    Let a be the number of trees in the grove [[var a]]. We're given [[eq a = 15]]. 
    Let b be the number of trees Grove workers will plant [[var b]].
    Let c be the number of trees in the grove after the workers are done [[var c]]. We have [[eq c = a + b]]. We're given [[eq c = 21]].
    The answer is the value of b [[answer b]].
    Q: {question}
    Peano solution:
    '''.strip() + '\n\n\n'


    # Evaluate math
    pi = 3.14
    if request.method == 'POST':

        # taking user input
        user_input = request.POST.get("question", "")
        user_input = str(user_input).lower()

        try:
            solution = (
                eval(user_input))  # example: 2 + 3 * 4 (any expression) (trigonometry functions not working)
            solution = str(solution)

            # plot
            plt.figure(figsize=(8, 1))
            plt.axhline(y=0, color='black', linewidth=2)
            plt.scatter(int(solution), 0, color='red', s=100, label=f'Number: {int(solution)}')
            plt.xlabel('Number Line')
            plt.yticks([])
            plt.legend()
            plt.savefig("output.jpg")

            # output
            # return JsonResponse({"answer": solution})  # 14
            evaluate_math = solution

        except:
            try:  # example: (5 * x + 4 = 14)   if the question is an equation
                symbol_dict = {}
                for i in user_input:
                    if i.isalpha():
                        symbol_dict[i] = sp.Symbol(i)
                symbol_list = list(symbol_dict.values())
                variable = symbol_list[0]

                lhs, rhs = user_input.split("=")
                lhs_expr = sp.sympify(lhs)
                rhs_expr = sp.sympify(rhs)
                equation1 = sp.Add(lhs_expr, -rhs_expr)
                solution = sp.solve(equation1, variable)

                # plot
                lhs, rhs = user_input.split("=")
                lhs_expr = sp.sympify(lhs)
                rhs_expr = sp.sympify(rhs)
                equation1 = sp.Add(lhs_expr, -rhs_expr)

                def equation(x):
                    return eval(str(equation1))

                x_values = np.linspace(-10, 10, 100)
                y_values = equation(x_values)

                plt.plot(x_values, y_values, label=user_input)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Equation Plot: {user_input}')
                plt.legend()
                plt.grid(True)
                plt.savefig("output.jpg")

                # output
                # return JsonResponse({"answer": str(solution[0])})  # 2
                evaluate_math = str(solution[0])


            except:
                try:  # for cubic or quadratic  #example: (x**2 + 2*x -15) if the question is quadratic equation
                    if ("**3" in user_input):
                        x = sp.Symbol('x')
                        equation_str = user_input
                        equation = sp.sympify(equation_str)
                        roots = sp.solve(equation, x)

                        # output
                        # return JsonResponse({"answer": str(roots)})
                        evaluate_math = str(roots)

                    else:
                        expr = user_input
                        for i in expr:
                            if i.isalpha():
                                variable = sp.Symbol(i)
                        roots = sp.solve(expr, variable)

                        x = sp.Symbol('x')
                        equation = sp.sympify(expr)
                        coefficients_dict = sp.collect(equation, x, evaluate=False)

                        # Fetch the coefficients
                        a = coefficients_dict[x ** 2]
                        b = coefficients_dict[x]
                        c = coefficients_dict[1]

                        print_str = ''
                        print_str += "STEP1: To calculate discriminant: b^2 - 4ac"
                        discriminant = b ** 2 - 4 * a * c
                        print_str += (f"discriminant: {discriminant}\n")

                        # Step 2: Check if the discriminant is positive, negative, or zero
                        if discriminant > 0:
                            print_str += (
                                "STEP2: Since the discriminant is positive, there are two distinct real roots.\n")
                        elif discriminant < 0:
                            print_str += ("STEP2: since the discriminant is negative,there are no real roots")
                        elif discriminant == 0:
                            print_str += (
                                "STEP2: since the discriminant is equal to zero, there is one repeated real root.\n")

                        print_str += "STEP3: Root1: (-b + sqrt(discriminant)) / (2*a)   and   Root2: (-b - sqrt(discriminant)) / (2*a)"

                        if roots != []:
                            roots_str = list(map(str, roots))
                            print_str += ("\n Roots are : " + roots_str[0] + ", " + roots_str[1])

                            # output
                            # return JsonResponse({"answer": print_str})
                            evaluate_math = print_str

                        else:
                            root1 = (-b + math.sqrt(discriminant)) / (2 * a)
                            root2 = (-b - math.sqrt(discriminant)) / (2 * a)
                            root1 = str(root1)
                            root2 = str(root2)
                            both_roots = root1 + root2

                            # plot
                            equation1 = user_input
                            x = sp.Symbol('x')
                            polynomial = sp.Poly(equation1, x)
                            coefficients = polynomial.all_coeffs()
                            a, b, c = coefficients
                            x = np.linspace(-10, 10, 100)
                            y = a * x ** 2 + b * x + c
                            plt.plot(x, y)
                            plt.xlabel('x-axis')
                            plt.ylabel('y-axis')
                            plt.title('Given Quadratic Equation')
                            plt.grid(True)
                            plt.savefig("output.jpg")

                            # output
                            # return JsonResponse({"answer": both_roots})  # "answer": ["-5","3"]
                            evaluate_math = both_roots
                except:
                    prompt = prompt_template.format(question = user_input)
                    second_completion = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    temperature = 0,
                    messages=[{"role": "user", "content": prompt }])
                    symbol = second_completion.choices[0].message.content
                    math_prompts = 'Analyze the given input, {symbolic_steps}. Provide the solution for the given input. You should also explain the soultion in precise, concise and step by step approach to the solution'
                    input_prompt = math_prompts.format(symbolic_steps = symbol)
                    Third_completion = openai.ChatCompletion.create(
                            model = "gpt-3.5-turbo",
                            temperature = 0,
                            messages=[
                            {"role": "system", "content": 'you are an expert maths symbolic interpreter and AI tutor who provide solution to given input and explains the solution' },
                            {"role": "user", "content": input_prompt}])
                        #print('Answer from except part')
                    evaluate_math = str(Third_completion.choices[0].message.content)





                 
                        

    # Category
    if request.method == 'POST':

        # input
        user_input = request.POST.get("question", "")  # example:(x**2 + 2*x -15)  any question
        user_input = str(user_input).lower()

        l = user_input.split(" ")
        category_list = []

        # equation type
        alpha_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't',
                            'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                            'M', 'N',
                            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        if 'x**3' in l or 'y**3' in l or 'x^3' in l or 'y^3' in l:
            category_list.append("Cubic Equation")
        elif 'x**2' in l or 'y**2' in l or 'x^2' in l or 'y^2' in l:
            category_list.append("Quadratic Equation")
        else:
            for i in alpha_letters:
                if i in l:
                    category_list.append("Polynomial Equation")

        # Number System
        number_system_key = ['number', 'numeral', '+', '-', '/', '*', 'numeration', 'place_value', 'compare',
                                "Natural_Numbers",
                                "natural", "whole", "integers", "rational", "irrational", "place", "face", "expanded",
                                "roman", "number", "estimation", "fractions", "decimal", "equivalent",
                                "simplification",
                                "addition", 'add', "subtraction", "multiplication", "division", "operations",
                                "terminating",
                                "non-terminating", "remainder", "line", "conversion", "exponents", "power", "square",
                                "cube", "real",
                                "laws", "rationalizing", "euclid's", "hcf", "lcm", 'pow', "euclidean", "fundamental"]

        for i in number_system_key:
            if i in l:
                category_list.append("number system")
            else:
                pass

        # Algebra
        algebra_key = ['algebra', 'expression', 'equation', "=", "algebraic", "expression", "equation", "variable",
                        "constant", "coefficient", "linear", "quadratic", "**", "polynomial", "factor", "identity",
                        'root',
                        "simultaneous", "inequality", "algebra", "formula", "exponent", 'log', "binomial",
                        "trinomial",
                        "equation", "simplification", "graph", "system"]

        for i in algebra_key:
            if i in l:
                category_list.append("algebra")
            else:
                pass

        # Mensuration
        mensuration_key = ['geometry', 'shape', 'area', 'perimeter', 'volume', "area", "perimeter", "rectangle",
                            "square", "triangle", "hypotenuse", "circle", "parallelogram", "rhombus", "trapezium",
                            "quadrilateral", "polygon",
                            "volume", "cylinder", "cone", "sphere", "cube", "cuboid", "pyramid", "hemisphere",
                            "surface",
                            "base",
                            "height", "radius", "diameter", "perpendicular", "diagonal", "isometric", "net",
                            "height",
                            "lateral", "surface", "right", "angled", "equilateral", "isosceles", "scalene", "obtuse",
                            "acute", "right-angled", "similar", "congruent", "prism", "composite",
                            "regular", "irregular", "symmetry", "axis", "point", "centroid", "circumcenter",
                            "incenter",
                            "orthocenter", "area", "perimeter"]

        for i in mensuration_key:
            if i in l:
                category_list.append("mensuration")
            else:
                pass

        # Statistics and Probability

        stat_prob_key = ['data', 'graph', 'chart', 'mean', 'median', 'mode', 'probability', 'chance', 'likely',
                            'unlikely',
                            "data", "mean", "median", "mode", "range", "frequency", "bar", "chart", "pie", "graph",
                            "probability",
                            "experiment", "outcomes", "sample", "event", "likely", "unlikely", "fair", "biased",
                            "tree",
                            "diagram", "combinations", "permutations", "random", "variable", "independent",
                            "dependent", "expected", "ratio", "odds", "experimental", "theoretical", "trials",
                            "success",
                            "failure", "mutually", "exclusive", "complementary", "union", "intersection",
                            "outcome", "probability", "distribution", "range", "mode", "mean", "median", "variance",
                            "standard",
                            "deviation", "percentile", "correlation", "regression", "sample", "population",
                            "confidence",
                            "interval", "hypothesis", "test"]

        for i in stat_prob_key:
            if i in l:
                category_list.append("statistics and probability")
            else:
                pass

        # Trigonometry
        trigonometry_key = ['trigonometry', "sin", "cos", "tan", "cot", "sec", "cosec", 'angle', 'triangle', 'sine',
                            'cosine', 'tangent',
                            "angle", "degree", "radian", "triangle", "right", "acute", "obtuse", "opposite",
                            "adjacent", "sine", "cosine", "tangent", "cotangent", "secant",
                            "cosecant", "pythagorean", "trigonometric", "identity", "solving", "equations",
                            "trigonometric", "ratios", "trigonometric", "functions", "angle", "measurements",
                            "angle", "sum", "difference", "multiple", "division", "values", "unit", "circle",
                            "solving",
                            "problems"]

        for i in trigonometry_key:
            if i in l:
                category_list.append('trigonometry')
            else:
                pass

        category_list = set(category_list)
        print_str = "the following question involves: "

        # output
        # return JsonResponse({"statement": print_str, "answer": str(
        #     category_list)})  # "the following question involves: ",  "{'Quadratic Equation', 'number system'}"
        category = str(print_str) + str(category_list)


    if request.method == 'POST':

        # input
        user_input = request.POST.get("question", "")  # example: (x + 12 = 20) any question
        user_input = str(user_input).lower()

        import random
        import linecache
        l = user_input.split(" ")
        category_list = []

        # equation type
        alpha_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't',
                            'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                            'M', 'N',
                            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        if 'x**3' in l or 'y**3' in l or 'x^3' in l or 'y^3' in l:
            category_list.append("Cubic Equation")
        elif 'x**2' in l or 'y**2' in l or 'x^2' in l or 'y^2' in l:
            category_list.append("Quadratic Equation")
        else:
            for i in alpha_letters:
                if i in l:
                    category_list.append("Polynomial Equation")

        # Number System
        number_system_key = ['number', 'numeral', '+', '-', '/', '*', 'numeration', 'place_value', 'compare',
                                "Natural_Numbers",
                                "natural", "whole", "integers", "rational", "irrational", "place", "face", "expanded",
                                "roman", "number", "estimation", "fractions", "decimal", "equivalent",
                                "simplification",
                                "addition", 'add', "subtraction", "multiplication", "division", "operations",
                                "terminating",
                                "non-terminating", "remainder", "line", "conversion", "exponents", "power", "square",
                                "cube", "real",
                                "laws", "rationalizing", "euclid's", "hcf", "lcm", 'pow', "euclidean", "fundamental"]

        for i in number_system_key:
            if i in l:
                category_list.append("number system")
            else:
                pass

        # Algebra
        algebra_key = ['algebra', 'expression', 'equation', "=", "algebraic", "expression", "equation", "variable",
                        "constant", "coefficient", "linear", "quadratic", "**", "polynomial", "factor", "identity",
                        'root',
                        "simultaneous", "inequality", "algebra", "formula", "exponent", 'log', "binomial",
                        "trinomial",
                        "equation", "simplification", "graph", "system"]

        for i in algebra_key:
            if i in l:
                category_list.append("algebra")
            else:
                pass

        # Mensuration
        mensuration_key = ['geometry', 'shape', 'area', 'perimeter', 'volume', "area", "perimeter", "rectangle",
                            "square", "triangle", "hypotenuse", "circle", "parallelogram", "rhombus", "trapezium",
                            "quadrilateral", "polygon",
                            "volume", "cylinder", "cone", "sphere", "cube", "cuboid", "pyramid", "hemisphere",
                            "surface",
                            "base",
                            "height", "radius", "diameter", "perpendicular", "diagonal", "isometric", "net",
                            "height",
                            "lateral", "surface", "right", "angled", "equilateral", "isosceles", "scalene", "obtuse",
                            "acute", "right-angled", "similar", "congruent", "prism", "composite",
                            "regular", "irregular", "symmetry", "axis", "point", "centroid", "circumcenter",
                            "incenter",
                            "orthocenter", "area", "perimeter"]

        for i in mensuration_key:
            if i in l:
                category_list.append("mensuration")
            else:
                pass

        # Statistics and Probability

        stat_prob_key = ['data', 'graph', 'chart', 'mean', 'median', 'mode', 'probability', 'chance', 'likely',
                            'unlikely',
                            "data", "mean", "median", "mode", "range", "frequency", "bar", "chart", "pie", "graph",
                            "probability",
                            "experiment", "outcomes", "sample", "event", "likely", "unlikely", "fair", "biased",
                            "tree",
                            "diagram", "combinations", "permutations", "random", "variable", "independent",
                            "dependent", "expected", "ratio", "odds", "experimental", "theoretical", "trials",
                            "success",
                            "failure", "mutually", "exclusive", "complementary", "union", "intersection",
                            "outcome", "probability", "distribution", "range", "mode", "mean", "median", "variance",
                            "standard",
                            "deviation", "percentile", "correlation", "regression", "sample", "population",
                            "confidence",
                            "interval", "hypothesis", "test"]

        for i in stat_prob_key:
            if i in l:
                category_list.append("statistics and probability")
            else:
                pass

        # Trigonometry
        trigonometry_key = ['trigonometry', "sin", "cos", "tan", "cot", "sec", "cosec", 'angle', 'triangle', 'sine',
                            'cosine', 'tangent',
                            "angle", "degree", "radian", "triangle", "right", "acute", "obtuse", "opposite",
                            "adjacent", "sine", "cosine", "tangent", "cotangent", "secant",
                            "cosecant", "pythagorean", "trigonometric", "identity", "solving", "equations",
                            "trigonometric", "ratios", "trigonometric", "functions", "angle", "measurements",
                            "angle", "sum", "difference", "multiple", "division", "values", "unit", "circle",
                            "solving",
                            "problems"]

        for i in trigonometry_key:
            if i in l:
                category_list.append('trigonometry')
            else:
                pass

        category_list = set(category_list)
        print(category_list)

        real_line = ''
        # Quadratic Equation  and Cubic Equation questions from 1 to 19
        if "Quadratic Equation" in category_list or "Cubic Equation" in category_list:
            que_num = random.sample(range(1, 19), 1)
            file = open('.//home//real_life_use.txt')
            content = file.readlines()
            real_line += (content[que_num[0]])

        # trigonometry questions from 50 to 61
        if "trigonometry" in category_list:
            que_num = random.sample(range(50, 61), 1)
            file = open('.//home//real_life_use.txt')
            content = file.readlines()
            real_line += (content[que_num[0]])

        # Algebra questions from 78 to 89
        if "algebra" in category_list:
            que_num = random.sample(range(78, 89), 1)
            file = open('.//home//real_life_use.txt')
            content = file.readlines()
            real_line += (content[que_num[0]])

        # number system questions from 66 to 77
        if "number system" in category_list:
            que_num = random.sample(range(66, 77), 1)
            file = open('.//home//real_life_use.txt')
            content = file.readlines()
            real_line += (content[que_num[0]])

        # mensuration questions from 22 to 33
        if "mensuration" in category_list:
            que_num = random.sample(range(22, 33), 1)
            file = open('.//home//real_life_use.txt')
            content = file.readlines()
            real_line += (content[que_num[0]])

        # probability and statistics from 36 ,47
        if "statistics and probability" in category_list:
            que_num = random.sample(range(36, 47), 1)
            file = open('.//home//real_life_use.txt')
            content = file.readlines()
            real_line += (content[que_num[0]])

        # output
        # return JsonResponse({"answer": real_line})
        real_life_examples = real_line


    # Level 1 qtns
    if request.method == 'POST':

        # input
        user_input = request.POST.get("question", "")  # example ( x + 5 = 10) any question
        user_input = str(user_input).lower()

        input_list = list(user_input)
        update_list = list()

        for i in input_list:
            if i.isdigit():
                try:
                    change = random.randint(1, int(i))
                except:
                    change = random.randint(0, int(i))

                update_list.append(str(change))
            else:
                update_list.append(i)

        new = ''.join(update_list)

        # output
        # return JsonResponse({"answer": str(new)})  # x + 3 = 7
        level_1_questions = str(new)
        


# Level 2 qtns
    if request.method == 'POST':

        # input
        user_input = request.POST.get("question", "")  # example: (x**2 + 2*x - 15) any question
        user_input = str(user_input).lower()

        import random
        import linecache

        l = user_input.split(" ")
        category_list = []

        # equation type
        alpha_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't',
                            'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                            'M', 'N',
                            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        if 'x**3' in l or 'y**3' in l or 'x^3' in l or 'y^3' in l:
            category_list.append("Cubic Equation")
        elif 'x**2' in l or 'y**2' in l or 'x^2' in l or 'y^2' in l:
            category_list.append("Quadratic Equation")
        else:
            for i in alpha_letters:
                if i in l:
                    category_list.append("Polynomial Equation")

        # Number System
        number_system_key = ['number', 'numeral', '+', '-', '/', '*', 'numeration', 'place_value', 'compare',
                                "Natural_Numbers",
                                "natural", "whole", "integers", "rational", "irrational", "place", "face", "expanded",
                                "roman", "number", "estimation", "fractions", "decimal", "equivalent",
                                "simplification",
                                "addition", 'add', "subtraction", "multiplication", "division", "operations",
                                "terminating",
                                "non-terminating", "remainder", "line", "conversion", "exponents", "power", "square",
                                "cube", "real",
                                "laws", "rationalizing", "euclid's", "hcf", "lcm", 'pow', "euclidean", "fundamental"]

        for i in number_system_key:
            if i in l:
                category_list.append("number system")
            else:
                pass

        # Algebra
        algebra_key = ['algebra', 'expression', 'equation', "=", "algebraic", "expression", "equation", "variable",
                        "constant", "coefficient", "linear", "quadratic", "**", "polynomial", "factor", "identity",
                        'root',
                        "simultaneous", "inequality", "algebra", "formula", "exponent", 'log', "binomial",
                        "trinomial",
                        "equation", "simplification", "graph", "system"]

        for i in algebra_key:
            if i in l:
                category_list.append("algebra")
            else:
                pass

        # Mensuration
        mensuration_key = ['geometry', 'shape', 'area', 'perimeter', 'volume', "area", "perimeter", "rectangle",
                            "square", "triangle", "hypotenuse", "circle", "parallelogram", "rhombus", "trapezium",
                            "quadrilateral", "polygon",
                            "volume", "cylinder", "cone", "sphere", "cube", "cuboid", "pyramid", "hemisphere",
                            "surface",
                            "base",
                            "height", "radius", "diameter", "perpendicular", "diagonal", "isometric", "net",
                            "height",
                            "lateral", "surface", "right", "angled", "equilateral", "isosceles", "scalene", "obtuse",
                            "acute", "right-angled", "similar", "congruent", "prism", "composite",
                            "regular", "irregular", "symmetry", "axis", "point", "centroid", "circumcenter",
                            "incenter",
                            "orthocenter", "area", "perimeter"]

        for i in mensuration_key:
            if i in l:
                category_list.append("mensuration")
            else:
                pass

        # Statistics and Probability

        stat_prob_key = ['data', 'graph', 'chart', 'mean', 'median', 'mode', 'probability', 'chance', 'likely',
                            'unlikely',
                            "data", "mean", "median", "mode", "range", "frequency", "bar", "chart", "pie", "graph",
                            "probability",
                            "experiment", "outcomes", "sample", "event", "likely", "unlikely", "fair", "biased",
                            "tree",
                            "diagram", "combinations", "permutations", "random", "variable", "independent",
                            "dependent", "expected", "ratio", "odds", "experimental", "theoretical", "trials",
                            "success",
                            "failure", "mutually", "exclusive", "complementary", "union", "intersection",
                            "outcome", "probability", "distribution", "range", "mode", "mean", "median", "variance",
                            "standard",
                            "deviation", "percentile", "correlation", "regression", "sample", "population",
                            "confidence",
                            "interval", "hypothesis", "test"]

        for i in stat_prob_key:
            if i in l:
                category_list.append("statistics and probability")
            else:
                pass

        # Trigonometry
        trigonometry_key = ['trigonometry', "sin", "cos", "tan", "cot", "sec", "cosec", 'angle', 'triangle', 'sine',
                            'cosine', 'tangent',
                            "angle", "degree", "radian", "triangle", "right", "acute", "obtuse", "opposite",
                            "adjacent", "sine", "cosine", "tangent", "cotangent", "secant",
                            "cosecant", "pythagorean", "trigonometric", "identity", "solving", "equations",
                            "trigonometric", "ratios", "trigonometric", "functions", "angle", "measurements",
                            "angle", "sum", "difference", "multiple", "division", "values", "unit", "circle",
                            "solving",
                            "problems"]

        for i in trigonometry_key:
            if i in l:
                category_list.append('trigonometry')
            else:
                pass

        category_list = set(category_list)

        # Quadratic Equation  and Cubic Equation questions from 174 to 200
        if "Quadratic Equation" in category_list or "Cubic Equation" in category_list:
            que_num = random.sample(range(174, 200), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_L2que = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_L2ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L2que += (content1[que_num[1]])
            print_L2ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L2que += (content1[que_num[2]])
            print_L2ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L2que, "answers": print_L2ans})
            level_2_questions = print_L2que + print_L2ans


        # trigonometry questions from 1 to 40
        elif "trigonometry" in category_list:
            que_num = random.sample(range(1, 41), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_L2que = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_L2ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L2que += (content1[que_num[1]])
            print_L2ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L2que += (content1[que_num[2]])
            print_L2ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L2que, "answers": print_L2ans})
            level_2_questions = print_L2que + print_L2ans


        # Algebra questions from 43 to 80
        elif "algebra" in category_list and "trigonometry" not in category_list:
            que_num = random.sample(range(46, 84), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_L2que = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_L2ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L2que += (content1[que_num[1]])
            print_L2ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L2que += (content1[que_num[2]])
            print_L2ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L2que, "answers": print_L2ans})
            level_2_questions = print_L2que + print_L2ans


        # number system questions from 87 to 123
        elif "number system" in category_list and "trigonometry" not in category_list and "algebra" not in category_list and "statistics and probability" not in category_list:
            que_num = random.sample(range(87, 123), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_L2que = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_L2ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L2que += (content1[que_num[1]])
            print_L2ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L2que += (content1[que_num[2]])
            print_L2ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L2que, "answers": print_L2ans})
            level_2_questions = print_L2que + print_L2ans


        # mensuration questions from 126 to 170
        elif "mensuration" in category_list and "trigonometry" not in category_list and "algebra" not in category_list and "statistics and probability" not in category_list:
            que_num = random.sample(range(126, 170), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_L2que = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_L2ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L2que += (content1[que_num[1]])
            print_L2ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L2que += (content1[que_num[2]])
            print_L2ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L2que, "answers": print_L2ans})
            level_2_questions = print_L2que + print_L2ans

            # statistics and probability questions from 204 to 221
        elif "statistics and probability" in category_list:
            que_num = random.sample(range(204, 221), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_L2que = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_L2ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L2que += (content1[que_num[1]])
            print_L2ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L2que += (content1[que_num[2]])
            print_L2ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L2que, "answers": print_L2ans})
            level_2_questions = print_L2que + print_L2ans

# Level 3 qtns
    if request.method == 'POST':

        # input
        user_input = request.POST.get("question", "")  # example: (x**2 + 2*x -15) any question
        user_input = str(user_input).lower()

        import random
        import linecache

        l = user_input.split(" ")
        category_list = []

        # equation type
        alpha_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't',
                            'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                            'M', 'N',
                            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        if 'x**3' in l or 'y**3' in l or 'x^3' in l or 'y^3' in l:
            category_list.append("Cubic Equation")
        elif 'x**2' in l or 'y**2' in l or 'x^2' in l or 'y^2' in l:
            category_list.append("Quadratic Equation")
        else:
            for i in alpha_letters:
                if i in l:
                    category_list.append("Polynomial Equation")

        # Number System
        number_system_key = ['number', 'numeral', '+', '-', '/', '*', 'numeration', 'place_value', 'compare',
                                "Natural_Numbers",
                                "natural", "whole", "integers", "rational", "irrational", "place", "face", "expanded",
                                "roman", "number", "estimation", "fractions", "decimal", "equivalent",
                                "simplification",
                                "addition", 'add', "subtraction", "multiplication", "division", "operations",
                                "terminating",
                                "non-terminating", "remainder", "line", "conversion", "exponents", "power", "square",
                                "cube", "real",
                                "laws", "rationalizing", "euclid's", "hcf", "lcm", 'pow', "euclidean", "fundamental"]

        for i in number_system_key:
            if i in l:
                category_list.append("number system")
            else:
                pass

        # Algebra
        algebra_key = ['algebra', 'expression', 'equation', "=", "algebraic", "expression", "equation", "variable",
                        "constant", "coefficient", "linear", "quadratic", "**", "polynomial", "factor", "identity",
                        'root',
                        "simultaneous", "inequality", "algebra", "formula", "exponent", 'log', "binomial",
                        "trinomial",
                        "equation", "simplification", "graph", "system"]

        for i in algebra_key:
            if i in l:
                category_list.append("algebra")
            else:
                pass

        # Mensuration
        mensuration_key = ['geometry', 'shape', 'area', 'perimeter', 'volume', "area", "perimeter", "rectangle",
                            "square", "triangle", "hypotenuse", "circle", "parallelogram", "rhombus", "trapezium",
                            "quadrilateral", "polygon",
                            "volume", "cylinder", "cone", "sphere", "cube", "cuboid", "pyramid", "hemisphere",
                            "surface",
                            "base",
                            "height", "radius", "diameter", "perpendicular", "diagonal", "isometric", "net",
                            "height",
                            "lateral", "surface", "right", "angled", "equilateral", "isosceles", "scalene", "obtuse",
                            "acute", "right-angled", "similar", "congruent", "prism", "composite",
                            "regular", "irregular", "symmetry", "axis", "point", "centroid", "circumcenter",
                            "incenter",
                            "orthocenter", "area", "perimeter"]

        for i in mensuration_key:
            if i in l:
                category_list.append("mensuration")
            else:
                pass

        # Statistics and Probability

        stat_prob_key = ['data', 'graph', 'chart', 'mean', 'median', 'mode', 'probability', 'chance', 'likely',
                            'unlikely',
                            "data", "mean", "median", "mode", "range", "frequency", "bar", "chart", "pie", "graph",
                            "probability",
                            "experiment", "outcomes", "sample", "event", "likely", "unlikely", "fair", "biased",
                            "tree",
                            "diagram", "combinations", "permutations", "random", "variable", "independent",
                            "dependent", "expected", "ratio", "odds", "experimental", "theoretical", "trials",
                            "success",
                            "failure", "mutually", "exclusive", "complementary", "union", "intersection",
                            "outcome", "probability", "distribution", "range", "mode", "mean", "median", "variance",
                            "standard",
                            "deviation", "percentile", "correlation", "regression", "sample", "population",
                            "confidence",
                            "interval", "hypothesis", "test"]

        for i in stat_prob_key:
            if i in l:
                category_list.append("statistics and probability")
            else:
                pass

        # Trigonometry
        trigonometry_key = ['trigonometry', "sin", "cos", "tan", "cot", "sec", "cosec", 'angle', 'triangle', 'sine',
                            'cosine', 'tangent',
                            "angle", "degree", "radian", "triangle", "right", "acute", "obtuse", "opposite",
                            "adjacent", "sine", "cosine", "tangent", "cotangent", "secant",
                            "cosecant", "pythagorean", "trigonometric", "identity", "solving", "equations",
                            "trigonometric", "ratios", "trigonometric", "functions", "angle", "measurements",
                            "angle", "sum", "difference", "multiple", "division", "values", "unit", "circle",
                            "solving",
                            "problems"]

        for i in trigonometry_key:
            if i in l:
                category_list.append('trigonometry')
            else:
                pass

        category_list = set(category_list)

        # Quadratic Equation  and Cubic Equation questions from 174 to 200
        if "Quadratic Equation" in category_list or "Cubic Equation" in category_list:
            que_num = random.sample(range(174, 200), 3)

            # question/ answer no.1
            file1 = open('.//home//L3questions.txt')
            content1 = file1.readlines()
            print_L3que = (content1[que_num[0]])

            file2 = open('.//home//L3answers.txt')
            content2 = file2.readlines()
            print_L3ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L3que += (content1[que_num[1]])
            print_L3ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L3que += (content1[que_num[2]])
            print_L3ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L3que, "answers": print_L3ans})
            level_3_questions = print_L3que + print_L3ans




        # trigonometry questions from 1 to 40
        elif "trigonometry" in category_list:
            que_num = random.sample(range(1, 41), 3)

            # question/ answer no.1
            file1 = open('.//home//L3questions.txt')
            content1 = file1.readlines()
            print_L3que = (content1[que_num[0]])

            file2 = open('.//home//L3answers.txt')
            content2 = file2.readlines()
            print_L3ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L3que += (content1[que_num[1]])
            print_L3ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L3que += (content1[que_num[2]])
            print_L3ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L3que, "answers": print_L3ans})
            level_3_questions = print_L3que + print_L3ans

        # Algebra questions from 43 to 80
        elif "algebra" in category_list and "trigonometry" not in category_list:
            que_num = random.sample(range(46, 84), 3)

            # question/ answer no.1
            file1 = open('.//home//L3questions.txt')
            content1 = file1.readlines()
            print_L3que = (content1[que_num[0]])

            file2 = open('.//home//L3answers.txt')
            content2 = file2.readlines()
            print_L3ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L3que += (content1[que_num[1]])
            print_L3ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L3que += (content1[que_num[2]])
            print_L3ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L3que, "answers": print_L3ans})
            level_3_questions = print_L3que + print_L3ans

        # number system questions from 87 to 123
        elif "number system" in category_list and "trigonometry" not in category_list and "algebra" not in category_list and "statistics and probability" not in category_list:
            que_num = random.sample(range(87, 123), 3)

            # question/ answer no.1
            file1 = open('.//home//L3questions.txt')
            content1 = file1.readlines()
            print_L3que = (content1[que_num[0]])

            file2 = open('.//home//L3answers.txt')
            content2 = file2.readlines()
            print_L3ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L3que += (content1[que_num[1]])
            print_L3ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L3que += (content1[que_num[2]])
            print_L3ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L3que, "answers": print_L3ans})
            level_3_questions = print_L3que + print_L3ans


        # mensuration questions from 126 to 170
        elif "mensuration" in category_list and "trigonometry" not in category_list and "algebra" not in category_list and "statistics and probability" not in category_list:
            que_num = random.sample(range(126, 170), 3)

            # question/ answer no.1
            file1 = open('.//home//L3questions.txt')
            content1 = file1.readlines()
            print_L3que = (content1[que_num[0]])

            file2 = open('.//home//L3answers.txt')
            content2 = file2.readlines()
            print_L3ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L3que += (content1[que_num[1]])
            print_L3ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L3que += (content1[que_num[2]])
            print_L3ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L3que, "answers": print_L3ans})
            level_3_questions = print_L3que + print_L3ans

        # statistics and probability questions from 204 to 221
        elif "statistics and probability" in category_list:
            que_num = random.sample(range(204, 221), 3)

            # question/ answer no.1
            file1 = open('.//home//L3questions.txt')
            content1 = file1.readlines()
            print_L3que = (content1[que_num[0]])

            file2 = open('.//home//L3answers.txt')
            content2 = file2.readlines()
            print_L3ans = (content2[que_num[0]])

            # question/ answer no.2
            print_L3que += (content1[que_num[1]])
            print_L3ans += (content2[que_num[1]])

            # question/ answer no.3
            print_L3que += (content1[que_num[2]])
            print_L3ans += (content2[que_num[2]])

            # output
            # return JsonResponse({"questions": print_L3que, "answers": print_L3ans})
            level_3_questions = print_L3que + print_L3ans

# Specific category questions -------------------------------------------------
    if request.method == 'POST':
        # input ; chose category from : Quadratic Equation, trigonometry, statistics and probability, mensuration, number system, algebra
        ask_category = request.POST.get("category", "")  # category from the above

        if ask_category == 'Quadratic Equation':
            que_num = random.sample(range(174, 201), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_Sque = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_Sans = (content2[que_num[0]])

            # question/ answer no.2
            print_Sque += (content1[que_num[1]])
            print_Sans += (content2[que_num[1]])

            # question/ answer no.3
            print_Sque += (content1[que_num[2]])
            print_Sans += (content2[que_num[2]])

            que_num2 = random.sample(range(174, 201), 2)

            # question/ answer no.4
            file3 = open('.//home//L3questions.txt')
            content3 = file3.readlines()
            print_Sque += (content3[que_num2[0]])

            file4 = open('.//home//L3answers.txt')
            content4 = file4.readlines()
            print_Sans += (content4[que_num2[0]])

            # question/ answer no.5
            print_Sque += (content3[que_num[1]])
            print_Sans += (content4[que_num[1]])

            # output
            # return JsonResponse({"questions": print_Sque, "answers": print_Sans})
            specific_category_questions = print_Sque + "\n" + print_Sans




        elif ask_category == 'statistics and probability':
            que_num = random.sample(range(204, 221), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_Sque = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_Sans = (content2[que_num[0]])

            # question/ answer no.2
            print_Sque += (content1[que_num[1]])
            print_Sans += (content2[que_num[1]])

            # question/ answer no.3
            print_Sque += (content1[que_num[2]])
            print_Sans += (content2[que_num[2]])

            que_num2 = random.sample(range(174, 201), 2)

            # question/ answer no.4
            file3 = open('.//home//L3questions.txt')
            content3 = file3.readlines()
            print_Sque += (content3[que_num2[0]])

            file4 = open('.//home//L3answers.txt')
            content4 = file4.readlines()
            print_Sans += (content4[que_num2[0]])

            # question/ answer no.5
            print_Sque += (content3[que_num[1]])
            print_Sans += (content4[que_num[1]])

            # output
            # return JsonResponse({"questions": print_Sque, "answers": print_Sans})
            specific_category_questions = print_Sque + "\n" + print_Sans



        elif ask_category == 'mensuration':
            que_num = random.sample(range(126, 171), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_Sque = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_Sans = (content2[que_num[0]])

            # question/ answer no.2
            print_Sque += (content1[que_num[1]])
            print_Sans += (content2[que_num[1]])

            # question/ answer no.3
            print_Sque += (content1[que_num[2]])
            print_Sans += (content2[que_num[2]])

            que_num2 = random.sample(range(174, 201), 2)

            # question/ answer no.4
            file3 = open('.//home//L3questions.txt')
            content3 = file3.readlines()
            print_Sque += (content3[que_num2[0]])

            file4 = open('.//home//L3answers.txt')
            content4 = file4.readlines()
            print_Sans += (content4[que_num2[0]])

            # question/ answer no.5
            print_Sque += (content3[que_num[1]])
            print_Sans += (content4[que_num[1]])

            # output
            # return JsonResponse({"questions": print_Sque, "answers": print_Sans})
            specific_category_questions = print_Sque + "\n" + print_Sans



        elif ask_category == 'number system':
            que_num = random.sample(range(87, 123), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_Sque = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_Sans = (content2[que_num[0]])

            # question/ answer no.2
            print_Sque += (content1[que_num[1]])
            print_Sans += (content2[que_num[1]])

            # question/ answer no.3
            print_Sque += (content1[que_num[2]])
            print_Sans += (content2[que_num[2]])

            que_num2 = random.sample(range(174, 201), 2)

            # question/ answer no.4
            file3 = open('.//home//L3questions.txt')
            content3 = file3.readlines()
            print_Sque += (content3[que_num2[0]])

            file4 = open('.//home//L3answers.txt')
            content4 = file4.readlines()
            print_Sans += (content4[que_num2[0]])

            # question/ answer no.5
            print_Sque += (content3[que_num[1]])
            print_Sans += (content4[que_num[1]])

            # output
            # return JsonResponse({"questions": print_Sque, "answers": print_Sans})
            specific_category_questions = print_Sque + "\n" + print_Sans



        elif ask_category == 'algebra':
            que_num = random.sample(range(46, 84), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_Sque = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_Sans = (content2[que_num[0]])

            # question/ answer no.2
            print_Sque += (content1[que_num[1]])
            print_Sans += (content2[que_num[1]])

            # question/ answer no.3
            print_Sque += (content1[que_num[2]])
            print_Sans += (content2[que_num[2]])

            que_num2 = random.sample(range(174, 201), 2)

            # question/ answer no.4
            file3 = open('.//home//L3questions.txt')
            content3 = file3.readlines()
            print_Sque += (content3[que_num2[0]])

            file4 = open('.//home//L3answers.txt')
            content4 = file4.readlines()
            print_Sans += (content4[que_num2[0]])

            # question/ answer no.5
            print_Sque += (content3[que_num[1]])
            print_Sans += (content4[que_num[1]])

            # output
            # return JsonResponse({"questions": print_Sque, "answers": print_Sans})
            specific_category_questions = print_Sque + "\n" + print_Sans



        elif ask_category == 'trigonometry':
            que_num = random.sample(range(1, 41), 3)

            # question/ answer no.1
            file1 = open('.//home//questions.txt')
            content1 = file1.readlines()
            print_Sque = (content1[que_num[0]])

            file2 = open('.//home//answers.txt')
            content2 = file2.readlines()
            print_Sans = (content2[que_num[0]])

            # question/ answer no.2
            print_Sque += (content1[que_num[1]])
            print_Sans += (content2[que_num[1]])

            # question/ answer no.3
            print_Sque += (content1[que_num[2]])
            print_Sans += (content2[que_num[2]])

            que_num2 = random.sample(range(174, 201), 2)

            # question/ answer no.4
            file3 = open('.//home//L3questions.txt')
            content3 = file3.readlines()
            print_Sque += (content3[que_num2[0]])

            file4 = open('.//home//L3answers.txt')
            content4 = file4.readlines()
            print_Sans += (content4[que_num2[0]])

            # question/ answer no.5
            print_Sque += (content3[que_num[1]])
            print_Sans += (content4[que_num[1]])

            # output
            # return JsonResponse({"questions": print_Sque, "answers": print_Sans})
            specific_category_questions = print_Sque + "\n" + print_Sans

    combined_data = {
        "math_data": evaluate_math,
        "category_data": category,
        "real_life_data": real_life_examples,
        "level_1_data": level_1_questions,
        "level_2_data": level_2_questions,
        "level_3_data": level_3_questions,
        "specific_category_data": specific_category_questions,
    }

    return JsonResponse(combined_data)
