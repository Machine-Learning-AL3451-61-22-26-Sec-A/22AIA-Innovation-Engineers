import streamlit as st

def calculate_square(number):
    return number ** 2

def main():
    st.title("Simple Square Calculator")

    # User input
    number = st.number_input("Enter a number:")

    # Calculate square
    square = calculate_square(number)

    # Display result
    st.write(f"The square of {number} is {square}")

if __name__ == "__main__":
    main()
