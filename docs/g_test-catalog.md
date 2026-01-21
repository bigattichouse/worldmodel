# Generated BluePrint Training Test Catalog Extension
**Date:** 2026-01-20  
**Purpose:** To supplement the original test catalog with new prompts and ideas, particularly for mathematical and scientific calculations.
**Reference:** Based on the methodology in `docs/blueprint-prompt.md`.

---

## Additions to Basic Systems

### Advanced Math & Physics Calculations (New Category)

This category focuses on more complex mathematical and scientific formulas that still represent self-contained, stateless calculations.

1.  **Quadratic Equation Solver:** A service to find the roots (x-intercepts) of a quadratic equation (`ax^2 + bx + c = 0`).
2.  **Matrix Multiplication Service:** A service that multiplies two matrices and returns the resulting matrix.
3.  **Vector Operations Service:** A service to perform common vector operations like dot product, cross product, and magnitude calculation.
4.  **Statistical Calculator:** A service that takes a list of numbers and calculates the mean, median, mode, and standard deviation.
5.  **Prime Number Generator:** A service to generate a list of prime numbers up to a specified limit.
6.  **Fibonacci Sequence Generator:** A service to generate the Fibonacci sequence up to the Nth term.
7.  **Projectile Motion Calculator:** A service to calculate the trajectory, maximum height, and range of a projectile given an initial velocity and angle.
8.  **Kinetic Energy Calculator:** A service to calculate the kinetic energy of an object given its mass and velocity.
9.  **Ohm's Law Calculator:** A service that can solve for voltage, current, or resistance given the other two values, based on Ohm's Law (V=IR).
10. **Simple Interest Calculator:** A service to calculate simple interest given principal, rate, and time.
11. **Area and Perimeter Calculator:** A service that can calculate the area and perimeter of various 2D shapes (circle, square, rectangle, triangle).
12. **Volume Calculator:** A service that can calculate the volume of various 3D shapes (cube, sphere, cylinder, cone).
13. **Factorial Calculator:** A service to compute the factorial of a non-negative integer.
14. **Logarithm Calculator:** A service to calculate logarithms for a given number, base, and type (e.g., natural log, log10).
15. **Relativistic Energy Calculator:** A service to calculate energy based on Einstein's `E=mc^2` formula.

### Additional Simple Calculation Ideas

1.  **Sales Tax Lookup Service:** Given a postal code, return the applicable sales tax rate.
2.  **Mortgage Affordability Calculator:** Estimate the maximum mortgage a person can afford based on their income, debts, and down payment.
3.  **Data Transfer Time Calculator:** Estimate the time required to transfer a file of a given size over a network with a given bandwidth.
4.  **Fuel Cost Calculator:** Calculate the total fuel cost for a trip given distance, fuel efficiency (MPG/KPL), and price per unit of fuel.
5.  **Kitchen Measurement Converter:** A service to convert between cooking units like cups, tablespoons, teaspoons, ounces, and milliliters.

### Very Basic Arithmetic Operations (New Subsection)

These prompts focus on fundamental arithmetic, which are foundational for more complex calculations.

1.  **Addition:** What is the sum of X and Y? (e.g., "What is 5 plus 3?", "Sum of 10 and 20")
2.  **Subtraction:** What is X minus Y? (e.g., "Subtract 7 from 15", "100 minus 40")
3.  **Multiplication:** What is X times Y? (e.g., "Multiply 6 by 8", "Triple X", "Double X")
4.  **Division:** What is X divided by Y? (e.g., "Divide 50 by 10", "Half of Z")
5.  **Square:** What is the square of X? (e.g., "What is 7 squared?")
6.  **Cube:** What is the cube of X? (e.g., "What is 3 cubed?")
7.  **Absolute Value:** What is the absolute value of X? (e.g., "Absolute value of -10")
8.  **Modulo:** What is the remainder when X is divided by Y? (e.g., "17 mod 5")
9.  **Power:** What is X raised to the power of Y? (e.g., "2 to the power of 3")

---

## Additions to Technical Systems

### Security & Cryptography

1.  **Hashing Service:** A service to generate a cryptographic hash (e.g., SHA-256, bcrypt) of a given string.
2.  **HMAC Generation Service:** A service to generate a Hash-based Message Authentication Code (HMAC) for a message and a secret key.
3.  **JWT (JSON Web Token) Validator:** A service to decode and validate a JWT, checking its signature, expiration, and claims.
4.  **TOTP (Time-based One-Time Password) Generator/Validator:** A service to generate and validate TOTPs, as used in multi-factor authentication.
5.  **Data Encryption/Decryption Service:** A service that provides methods to encrypt and decrypt data using symmetric (e.g., AES) or asymmetric (e.g., RSA) encryption.

### Networking

1.  **Subnet Calculator:** A service that takes an IP address and a subnet mask and calculates the network address, broadcast address, and number of available hosts.
2.  **DNS Lookup Service:** A service that performs DNS lookups (e.g., A, CNAME, MX records) for a given domain name.
3.  **Ping/Health Check Service:** A service that checks the status of a given IP address or hostname by attempting to connect to a specific port.

---
This document serves as a source for future dataset generation tasks.
