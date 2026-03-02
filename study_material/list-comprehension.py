# 1. Print All Elements
def print_all_elements(arr):
    [print(x) for x in arr]


# 2. Find Sum of Array
def find_sum(arr):
    return sum([x for x in arr])


# 3. Find Average
def find_average(arr):
    return sum([x for x in arr]) / len(arr) if arr else 0


# 4. Find Largest Element
def find_largest(arr):
    return max([x for x in arr]) if arr else None


# 5. Find Smallest Element
def find_smallest(arr):
    return min([x for x in arr]) if arr else None


# 6. Search Element
def search_element(arr, k):
    return any([x == k for x in arr])


# 7. Count Even and Odd
def count_even_odd(arr):
    return (
        len([x for x in arr if x % 2 == 0]),
        len([x for x in arr if x % 2 != 0])
    )


# 8. Check if Sorted
def is_sorted(arr):
    return all([arr[i] <= arr[i+1] for i in range(len(arr)-1)])


# 9. Find Index
def find_index(arr, k):
    return next((i for i in range(len(arr)) if arr[i] == k), -1)


# 10. Copy Array
def copy_array(arr):
    return [x for x in arr]


# 11. Print Reverse
def print_reverse(arr):
    [print(arr[i]) for i in range(len(arr)-1, -1, -1)]


# 12. Min Adjacent Difference
def min_adjacent_diff(arr):
    return min([abs(arr[i+1]-arr[i]) for i in range(len(arr)-1)]) if len(arr) > 1 else 0


# 13. Contains Zero
def contains_zero(arr):
    return any([x == 0 for x in arr])


# 14. Count Greater Than K
def count_greater_than_k(arr, k):
    return len([x for x in arr if x > k])


# 15. Initialize Array
def initialize_array(n, val):
    return [val for _ in range(n)]


# 16. Print Every Second
def print_every_second(arr):
    [print(arr[i]) for i in range(0, len(arr), 2)]


# 17. Replace Negatives
def replace_negatives(arr):
    return [0 if x < 0 else x for x in arr]


# 18. Contains Number
def contains_number(arr, k):
    return any([x == k for x in arr])


# 19. First Element
def get_first_element(arr):
    return arr[0] if arr else None


# 20. Middle Element
def get_middle_element(arr):
    return arr[len(arr)//2] if arr else None


# 21. Last Element
def get_last_element(arr):
    return arr[-1] if arr else None


# 22. Compare First & Last
def compare_first_last(arr):
    return arr[0] == arr[-1] if arr else False


# 23. Sum Even Indices
def sum_even_indices(arr):
    return sum([arr[i] for i in range(0, len(arr), 2)])


# 24. Multiply by K
def multiply_by_k(arr, k):
    return [x*k for x in arr]


# 25. Count Positives
def count_positives(arr):
    return len([x for x in arr if x > 0])


# 26. All Elements Same
def all_elements_same(arr):
    return len(set([x for x in arr])) <= 1


# 27. Print Reverse Order
def print_reverse_order(arr):
    [print(x) for x in arr[::-1]]


# 28. Find Product
def find_product(arr):
    from functools import reduce
    return reduce(lambda a,b: a*b, [x for x in arr], 1) if arr else 0


# 29. Is Array Empty
def is_array_empty(arr):
    return len([x for x in arr]) == 0


# 30. Create Squares
def create_squares_array(arr):
    return [x*x for x in arr]

# 1. Count Negative Numbers
def count_negatives(arr):
    return len([x for x in arr if x < 0])


# 2. Get Even Numbers
def get_evens(arr):
    return [x for x in arr if x % 2 == 0]


# 3. Get Odd Numbers
def get_odds(arr):
    return [x for x in arr if x % 2 != 0]


# 4. Double Each Element
def double_elements(arr):
    return [x*2 for x in arr]


# 5. Cube Each Element
def cube_elements(arr):
    return [x**3 for x in arr]


# 6. Absolute Values
def absolute_values(arr):
    return [abs(x) for x in arr]


# 7. Filter Strings Longer Than 3
def long_strings(arr):
    return [s for s in arr if len(s) > 3]


# 8. Length of Each String
def string_lengths(arr):
    return [len(s) for s in arr]


# 9. Remove Empty Strings
def remove_empty_strings(arr):
    return [s for s in arr if s != ""]


# 10. Square Only Even Numbers
def square_evens(arr):
    return [x*x for x in arr if x % 2 == 0]


# 11. Flatten 2D List
def flatten_2d(arr):
    return [x for row in arr for x in row]


# 12. Matrix Transpose
def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


# 13. Pair Elements with Index
def index_pairs(arr):
    return [(i, arr[i]) for i in range(len(arr))]


# 14. Reverse Strings
def reverse_strings(arr):
    return [s[::-1] for s in arr]


# 15. Extract First Letters
def first_letters(arr):
    return [s[0] for s in arr if s]


# 16. Remove Duplicates (Preserve Order)
def remove_duplicates(arr):
    seen = []
    return [seen.append(x) or x for x in arr if x not in seen]


# 17. Multiply Corresponding Elements
def multiply_lists(a, b):
    return [a[i]*b[i] for i in range(min(len(a), len(b)))]


# 18. Get Divisible by K
def divisible_by_k(arr, k):
    return [x for x in arr if x % k == 0]


# 19. Capitalize Strings
def capitalize_strings(arr):
    return [s.capitalize() for s in arr]


# 20. Find Common Elements
def common_elements(a, b):
    return [x for x in a if x in b]


# 21. Difference of Lists
def difference(a, b):
    return [x for x in a if x not in b]


# 22. Running Sum
def running_sum(arr):
    return [sum(arr[:i+1]) for i in range(len(arr))]


# 23. All Positive?
def all_positive(arr):
    return all([x > 0 for x in arr])


# 24. Any Negative?
def any_negative(arr):
    return any([x < 0 for x in arr])


# 25. Count Occurrences of K
def count_occurrences(arr, k):
    return len([x for x in arr if x == k])


# 26. Create List of Tuples (x, x^2)
def number_square_pairs(arr):
    return [(x, x*x) for x in arr]


# 27. Replace Odd with -1
def replace_odd(arr):
    return [-1 if x % 2 != 0 else x for x in arr]


# 28. Filter Palindromes (Strings)
def palindrome_strings(arr):
    return [s for s in arr if s == s[::-1]]


# 29. Multiply Table of N
def multiplication_table(n):
    return [n*i for i in range(1, 11)]


# 30. Generate All Pairs
def all_pairs(arr):
    return [(x, y) for x in arr for y in arr]

# 1. Elements Greater Than Average
def greater_than_average(arr):
    avg = sum(arr)/len(arr) if arr else 0
    return [x for x in arr if x > avg]


# 2. Square Until Limit (x^2 < 50)
def squares_below_50(arr):
    return [x*x for x in arr if x*x < 50]


# 3. Remove None Values
def remove_none(arr):
    return [x for x in arr if x is not None]


# 4. Convert Strings to Integers
def strings_to_ints(arr):
    return [int(x) for x in arr]


# 5. Extract Digits from String
def extract_digits(s):
    return [int(ch) for ch in s if ch.isdigit()]


# 6. Find All Factors of N
def factors(n):
    return [i for i in range(1, n+1) if n % i == 0]


# 7. Prime Numbers in List
def primes_in_list(arr):
    return [x for x in arr if x > 1 and all(x % i != 0 for i in range(2, int(x**0.5)+1))]


# 8. Remove Vowels from Strings
def remove_vowels(arr):
    return [''.join([ch for ch in s if ch.lower() not in 'aeiou']) for s in arr]


# 9. Count Words in Sentences
def word_counts(arr):
    return [len(s.split()) for s in arr]


# 10. Filter Numbers with Exactly 2 Digits
def two_digit_numbers(arr):
    return [x for x in arr if 10 <= abs(x) <= 99]


# 11. Pair Each Element with Its Square
def pair_with_square(arr):
    return [(x, x*x) for x in arr]


# 12. Generate Coordinate Grid (0 to n-1)
def coordinate_grid(n):
    return [(i, j) for i in range(n) for j in range(n)]


# 13. All Substrings of String
def all_substrings(s):
    return [s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)]


# 14. Elements at Prime Indices
def prime_index_elements(arr):
    primes = [i for i in range(len(arr)) if i > 1 and all(i % j != 0 for j in range(2, int(i**0.5)+1))]
    return [arr[i] for i in primes]


# 15. Convert Celsius to Fahrenheit
def c_to_f(arr):
    return [(c * 9/5) + 32 for c in arr]


# 16. Unique Characters in String
def unique_chars(s):
    return [ch for ch in s if s.count(ch) == 1]


# 17. Duplicate Each Character
def duplicate_chars(s):
    return ''.join([ch*2 for ch in s])


# 18. Remove Numbers from String
def remove_numbers(s):
    return ''.join([ch for ch in s if not ch.isdigit()])


# 19. Reverse Words in Sentence
def reverse_words(sentence):
    return ' '.join([word[::-1] for word in sentence.split()])


# 20. Intersection of Multiple Lists
def multi_intersection(*lists):
    return [x for x in lists[0] if all(x in lst for lst in lists)]


# 21. Count Uppercase Letters
def count_uppercase(s):
    return len([ch for ch in s if ch.isupper()])


# 22. Convert to Binary
def to_binary(arr):
    return [bin(x)[2:] for x in arr]


# 23. Running Product
def running_product(arr):
    return [eval('*'.join(map(str, arr[:i+1]))) for i in range(len(arr))]


# 24. Replace Multiples of 3 with 'Fizz'
def fizz_replace(arr):
    return ['Fizz' if x % 3 == 0 else x for x in arr]


# 25. Flatten 3D List
def flatten_3d(arr):
    return [z for x in arr for y in x for z in y]


# 26. Generate Fibonacci (n terms)
def fibonacci(n):
    return [0,1] if n==2 else [0,1] + [sum([0,1] + [sum([0,1])])]


# 27. Words Starting with Vowel
def starts_with_vowel(arr):
    return [s for s in arr if s[0].lower() in 'aeiou']


# 28. Length-Based Sorting Key List
def length_pairs(arr):
    return [(s, len(s)) for s in arr]


# 29. Find Duplicates in List
def find_duplicates(arr):
    return [x for x in arr if arr.count(x) > 1]


# 30. All Possible 3-Element Combinations
def combinations_of_three(arr):
    return [(arr[i], arr[j], arr[k])
            for i in range(len(arr))
            for j in range(i+1, len(arr))
            for k in range(j+1, len(arr))]


# 1. Elements Appearing Exactly Once
def appear_once(arr):
    return [x for x in arr if arr.count(x) == 1]


# 2. Elements Appearing More Than Twice
def appear_more_than_twice(arr):
    return [x for x in arr if arr.count(x) > 2]


# 3. Swap Case for All Strings
def swap_case(arr):
    return [s.swapcase() for s in arr]


# 4. Filter Perfect Squares
def perfect_squares(arr):
    return [x for x in arr if int(x**0.5)**2 == x]


# 5. Difference Between Adjacent Elements
def adjacent_differences(arr):
    return [arr[i+1] - arr[i] for i in range(len(arr)-1)]


# 6. Sum of Digits for Each Number
def sum_of_digits(arr):
    return [sum(int(d) for d in str(abs(x))) for x in arr]


# 7. Numbers Ending With 5
def ends_with_five(arr):
    return [x for x in arr if str(x).endswith('5')]


# 8. Remove Consecutive Duplicates
def remove_consecutive_duplicates(arr):
    return [arr[i] for i in range(len(arr)) if i == 0 or arr[i] != arr[i-1]]


# 9. Filter Armstrong Numbers
def armstrong_numbers(arr):
    return [x for x in arr if x == sum(int(d)**len(str(x)) for d in str(x))]


# 10. Reverse Only Even Numbers
def reverse_evens(arr):
    evens = [x for x in arr if x % 2 == 0][::-1]
    return [evens.pop(0) if x % 2 == 0 else x for x in arr]


# 11. Elements Smaller Than Next
def smaller_than_next(arr):
    return [arr[i] for i in range(len(arr)-1) if arr[i] < arr[i+1]]


# 12. Running Maximum
def running_max(arr):
    return [max(arr[:i+1]) for i in range(len(arr))]


# 13. All Divisor Pairs
def divisor_pairs(n):
    return [(i, n//i) for i in range(1, int(n**0.5)+1) if n % i == 0]


# 14. Palindrome Numbers
def palindrome_numbers(arr):
    return [x for x in arr if str(x) == str(x)[::-1]]


# 15. Remove Special Characters
def remove_special(s):
    return ''.join([ch for ch in s if ch.isalnum()])


# 16. All Even Index Elements Squared
def even_index_squared(arr):
    return [arr[i]**2 for i in range(0, len(arr), 2)]


# 17. Sum of Rows in Matrix
def row_sums(matrix):
    return [sum(row) for row in matrix]


# 18. Column Sums in Matrix
def column_sums(matrix):
    return [sum(row[i] for row in matrix) for i in range(len(matrix[0]))]


# 19. Remove Leading Zeros From Strings
def remove_leading_zeros(arr):
    return [s.lstrip('0') or '0' for s in arr]


# 20. Elements Greater Than Previous
def greater_than_previous(arr):
    return [arr[i] for i in range(1, len(arr)) if arr[i] > arr[i-1]]


# 21. Generate All Subarrays
def all_subarrays(arr):
    return [arr[i:j] for i in range(len(arr)) for j in range(i+1, len(arr)+1)]


# 22. Count Vowels in Each Word
def vowel_count(arr):
    return [sum(1 for ch in s.lower() if ch in 'aeiou') for s in arr]


# 23. Multiply Matrix by 2
def double_matrix(matrix):
    return [[x*2 for x in row] for row in matrix]


# 24. Filter Leap Years
def leap_years(arr):
    return [y for y in arr if (y%4==0 and y%100!=0) or (y%400==0)]


# 25. Pair Adjacent Elements
def adjacent_pairs(arr):
    return [(arr[i], arr[i+1]) for i in range(len(arr)-1)]


# 26. Reverse Words Order
def reverse_word_order(sentence):
    return ' '.join([word for word in sentence.split()][::-1])


# 27. Count Consonants in Each Word
def consonant_count(arr):
    return [sum(1 for ch in s.lower() if ch.isalpha() and ch not in 'aeiou') for s in arr]


# 28. All Elements Multiplied by Index
def multiply_by_index(arr):
    return [arr[i]*i for i in range(len(arr))]


# 29. Check Strictly Increasing
def strictly_increasing(arr):
    return all([arr[i] < arr[i+1] for i in range(len(arr)-1)])


# 30. All Unique Pairs (No Order Repeat)
def unique_pairs(arr):
    return [(arr[i], arr[j]) for i in range(len(arr)) for j in range(i+1, len(arr))]
