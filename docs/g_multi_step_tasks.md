# Generated Multi-Step and Logic Tasks for WorldModel Training
**Date:** 2026-01-21  
**Purpose:** To provide a set of simple tasks that require multiple steps or basic logic to solve. These tasks are more complex than single-shot questions but less complex than full system designs.
**Reference:** Inspired by `docs/g_test-catalog.md`.

---

## Multi-Step Calculations

This section contains problems that require a sequence of calculations to solve.

1.  **Circle Area from Diameter:** The diameter of a circle is 10 meters. What is its area? (Requires finding the radius first).
2.  **Item Cost with Sales Tax:** An item costs $50. The sales tax is 8%. What is the total cost?
3.  **Final Velocity:** A car starts at 10 m/s and accelerates at 2 m/s² for 5 seconds. What is its final velocity? (v_f = v_i + a*t)
4.  **Loan Monthly Payment:** Calculate the monthly payment for a $10,000 loan with a 5% annual interest rate over 3 years.
5.  **Age in Years and Months:** If a person was born on 1990-05-15 and today is 2026-01-21, how old are they in years and months?
6.  **Trip Fuel Cost:** A car has a fuel efficiency of 30 miles per gallon. If the trip is 150 miles and gas costs $3.50 per gallon, what is the total fuel cost?
7.  **Body Mass Index (BMI):** A person is 1.75 meters tall and weighs 70 kilograms. Calculate their BMI and determine if they are in the "normal" weight range (18.5-24.9).

## Simple Logic and Control Flow

This section contains tasks that require conditional logic (if/else) or simple loops.

1.  **Even or Odd:** Given the number 42, is it even or odd?
2.  **Number Filtering:** From the list [1, 5, 10, 15, 20, 25], find all numbers greater than 12.
3.  **Temperature-based Suggestion:** If the temperature is 10°C, would you suggest wearing a jacket?
4.  **Voting Eligibility:** A person is 17 years old. Are they eligible to vote in the United States?
5.  **FizzBuzz:** For the number 15, what is the FizzBuzz result? (Should be "FizzBuzz"). For the number 9? (Should be "Fizz").
6.  **Palindrome Check:** Is the word "racecar" a palindrome? What about "hello"?
7.  **Longest String:** From the list of strings ["apple", "banana", "kiwi"], which one is the longest?
8.  **Grade Calculator:** A student's score is 85 out of 100. What is their letter grade? (A > 90, B > 80, etc.)

## Simple Data Manipulation

This section involves tasks for processing and transforming small datasets.

1.  **Name Formatting:** Given the name "John Smith", format it as "Smith, John".
2.  **Word Count:** How many words are in the sentence "This is a simple sentence."?
3.  **Word Frequency:** In the text "apple banana apple orange banana apple", what is the frequency of each word?
4.  **Sort by Price:** Given a list of products `[{"name": "apple", "price": 0.5}, {"name": "banana", "price": 0.3}]`, sort them by price in ascending order.
5.  **Remove Duplicates:** From the list [1, 2, 2, 3, 4, 4, 5], create a new list with duplicates removed.
6.  **String to Uppercase:** Convert the string "hello world" to uppercase.
7.  **CSV Parsing:** Parse the CSV string "name,age\nJohn,30\nJane,25" into a list of lists.
8.  **Extract Vowels:** From the string "programming is fun", extract all the vowels.

## Simple HTML/DOM Interactions

This section contains conceptual tasks for interacting with a simple web page, expecting code snippets as answers.

1.  **Get Input Value:** Write JavaScript to get the value of an HTML input field with the ID "username".
2.  **Handle Button Click:** Write JavaScript to show an alert with the message "Button clicked!" when a button with the ID "submitBtn" is clicked.
3.  **Change Element Text:** Write JavaScript to change the text of a `<p>` element with the ID "message" to "Processing complete.".
4.  **Append List Item:** Write JavaScript to create a new `<li>` element with the text "New Item" and append it to a `<ul>` with the ID "itemList".
5.  **Toggle CSS Class:** Write JavaScript to add or remove the CSS class "is-active" on a `<div>` with the ID "menu" each time a button is clicked.
6.  **Read from Data Attribute:** An element has a `data-user-id="123"`. Write JavaScript to read this value.
7.  **Disable a Button:** Write JavaScript to disable a button with the ID "processBtn" after it has been clicked once.
8.  **Create an Image Element:** Write JavaScript to create an `<img>` element, set its `src` to "image.jpg", and append it to a `<div>` with the id "image-container".
