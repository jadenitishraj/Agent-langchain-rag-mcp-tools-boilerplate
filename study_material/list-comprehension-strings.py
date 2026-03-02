# 1. Convert All Strings to Uppercase
def to_upper(arr):
    return [s.upper() for s in arr]


# 2. Convert All Strings to Lowercase
def to_lower(arr):
    return [s.lower() for s in arr]


# 3. Reverse Each String
def reverse_each(arr):
    return [s[::-1] for s in arr]


# 4. Remove Spaces from Each String
def remove_spaces(arr):
    return [s.replace(" ", "") for s in arr]


# 5. Count Characters in Each String
def char_count(arr):
    return [len(s) for s in arr]


# 6. Keep Only Alphabetic Strings
def only_alpha(arr):
    return [s for s in arr if s.isalpha()]


# 7. Keep Only Numeric Strings
def only_numeric(arr):
    return [s for s in arr if s.isdigit()]


# 8. First and Last Character of Each String
def first_last_chars(arr):
    return [(s[0], s[-1]) for s in arr if s]


# 9. Remove Strings Shorter Than 5
def min_length_5(arr):
    return [s for s in arr if len(s) >= 5]


# 10. Duplicate Each String
def duplicate_strings(arr):
    return [s*2 for s in arr]


# 11. Remove All Vowels
def remove_vowels(arr):
    return [''.join([ch for ch in s if ch.lower() not in "aeiou"]) for s in arr]


# 12. Keep Strings Starting With Capital
def starts_capital(arr):
    return [s for s in arr if s and s[0].isupper()]


# 13. Count Words in Each String
def word_count(arr):
    return [len(s.split()) for s in arr]


# 14. Replace 'a' with '@'
def replace_a(arr):
    return [s.replace('a', '@') for s in arr]


# 15. Keep Palindrome Strings
def palindrome_strings(arr):
    return [s for s in arr if s == s[::-1]]


# 16. Extract All Digits from Each String
def extract_digits(arr):
    return [''.join([ch for ch in s if ch.isdigit()]) for s in arr]


# 17. Remove Special Characters
def remove_special(arr):
    return [''.join([ch for ch in s if ch.isalnum()]) for s in arr]


# 18. Capitalize First Letter
def capitalize_first(arr):
    return [s.capitalize() for s in arr]


# 19. Swap Case
def swap_case(arr):
    return [s.swapcase() for s in arr]


# 20. Reverse Word Order
def reverse_word_order(arr):
    return [' '.join(s.split()[::-1]) for s in arr]


# 21. Length of Longest Word in Each String
def longest_word_length(arr):
    return [max([len(word) for word in s.split()]) if s.split() else 0 for s in arr]


# 22. Keep Strings Containing Only Lowercase
def only_lowercase(arr):
    return [s for s in arr if s.islower()]


# 23. Add Prefix "Mr. "
def add_prefix(arr):
    return ["Mr. " + s for s in arr]


# 24. Add Suffix ".com"
def add_suffix(arr):
    return [s + ".com" for s in arr]


# 25. Remove Duplicate Characters in Each String
def remove_duplicate_chars(arr):
    return [''.join([ch for i, ch in enumerate(s) if ch not in s[:i]]) for s in arr]


# 26. Count Vowels in Each String
def count_vowels(arr):
    return [sum([1 for ch in s.lower() if ch in "aeiou"]) for s in arr]


# 27. Keep Strings Ending with 'ing'
def ends_with_ing(arr):
    return [s for s in arr if s.endswith("ing")]


# 28. Convert to Title Case
def title_case(arr):
    return [s.title() for s in arr]


# 29. Remove Strings Containing Numbers
def no_numbers(arr):
    return [s for s in arr if not any(ch.isdigit() for ch in s)]


# 30. Create (string, length) Pairs
def string_length_pairs(arr):
    return [(s, len(s)) for s in arr]


# 1. Remove Trailing Spaces
def strip_trailing(arr):
    return [s.rstrip() for s in arr]


# 2. Remove Leading Spaces
def strip_leading(arr):
    return [s.lstrip() for s in arr]


# 3. Strip Both Sides
def strip_both(arr):
    return [s.strip() for s in arr]


# 4. Count Spaces in Each String
def count_spaces(arr):
    return [s.count(" ") for s in arr]


# 5. Remove All Spaces
def remove_all_spaces(arr):
    return [''.join([ch for ch in s if ch != " "]) for s in arr]


# 6. Strings With More Than 2 Vowels
def more_than_two_vowels(arr):
    return [s for s in arr if sum(ch.lower() in "aeiou" for ch in s) > 2]


# 7. Extract Hashtags
def extract_hashtags(arr):
    return [[word for word in s.split() if word.startswith("#")] for s in arr]


# 8. Keep Strings With At Least One Digit
def has_digit(arr):
    return [s for s in arr if any(ch.isdigit() for ch in s)]


# 9. Reverse Only Words (Not Order)
def reverse_words_only(arr):
    return [' '.join([word[::-1] for word in s.split()]) for s in arr]


# 10. Replace Spaces With Underscore
def spaces_to_underscore(arr):
    return [s.replace(" ", "_") for s in arr]


# 11. Keep Strings of Even Length
def even_length_strings(arr):
    return [s for s in arr if len(s) % 2 == 0]


# 12. Repeat Each Character 3 Times
def triple_chars(arr):
    return [''.join([ch*3 for ch in s]) for s in arr]


# 13. Extract Only Uppercase Letters
def only_uppercase_chars(arr):
    return [''.join([ch for ch in s if ch.isupper()]) for s in arr]


# 14. Remove Punctuation
def remove_punctuation(arr):
    return [''.join([ch for ch in s if ch.isalnum() or ch.isspace()]) for s in arr]


# 15. Strings Starting and Ending Same
def start_end_same(arr):
    return [s for s in arr if s and s[0] == s[-1]]


# 16. Middle Character of Each String
def middle_char(arr):
    return [s[len(s)//2] if s else "" for s in arr]


# 17. Reverse Every Second String
def reverse_every_second(arr):
    return [s[::-1] if i % 2 else s for i, s in enumerate(arr)]


# 18. Count Consonants in Each String
def count_consonants(arr):
    return [sum(ch.isalpha() and ch.lower() not in "aeiou" for ch in s) for s in arr]


# 19. Remove Words Shorter Than 4
def remove_short_words(arr):
    return [' '.join([word for word in s.split() if len(word) >= 4]) for s in arr]


# 20. Extract Emails (Simple Check)
def extract_emails(arr):
    return [s for s in arr if "@" in s and "." in s]


# 21. Alternate Upper Lower Characters
def alternate_case(arr):
    return [''.join([ch.upper() if i%2==0 else ch.lower() for i, ch in enumerate(s)]) for s in arr]


# 22. Remove Duplicate Words
def remove_duplicate_words(arr):
    return [' '.join([word for i, word in enumerate(s.split()) if word not in s.split()[:i]]) for s in arr]


# 23. Count Each Word Length
def word_lengths(arr):
    return [[len(word) for word in s.split()] for s in arr]


# 24. Keep Strings Containing Substring "py"
def contains_py(arr):
    return [s for s in arr if "py" in s.lower()]


# 25. Reverse Entire String List Order
def reverse_list(arr):
    return [arr[i] for i in range(len(arr)-1, -1, -1)]


# 26. Extract First Word
def first_word(arr):
    return [s.split()[0] if s.split() else "" for s in arr]


# 27. Extract Last Word
def last_word(arr):
    return [s.split()[-1] if s.split() else "" for s in arr]


# 28. Strings Without Vowels
def no_vowel_strings(arr):
    return [s for s in arr if not any(ch.lower() in "aeiou" for ch in s)]


# 29. Replace Digits With '*'
def mask_digits(arr):
    return [''.join(['*' if ch.isdigit() else ch for ch in s]) for s in arr]


# 30. Create (word, reversed_word) Pairs
def word_reverse_pairs(arr):
    return [(s, s[::-1]) for s in arr]
