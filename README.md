# statlift
Free Analytics for Strong Data. :rocket:

# :mechanical_arm: About:

StatLift is a web app that enables users of the [Strong App](https://www.strong.app/) to keep track of their training progress. 

StatLift was built with [Streamlit](https://streamlit.io/) and is hosted on the Streamlit Community Cloud. 

Find it here: https://jjaju-statlift-statlift-amb2ux.streamlit.app/

I'm neither a Streamlit expert nor physically able to produce the most impressive dataset for stress testing StatLift, so feel free to leave feedback or suggestions for improvement by opening a new [issue](https://github.com/jjaju/statlift/issues).

# :bulb: How to use:

1. Export your workout data from the Strong App:

    > *Profile -> Settings -> Export Strong Data*

2. Visit https://jjaju-statlift-statlift-amb2ux.streamlit.app/

3. Upload your exported csv file and celebrate your training progress


# :computer: Alternatively run StatLift locally:

1. Clone this repository:

    `git clone https://github.com/jjaju/statlift.git`

2. Navigate to cloned folder:

    `cd statlift`

3. Install required packages:

    `pip install -r requirements.txt`

4. Start statlift using streamlit:

    `streamlit run statlift.py`
