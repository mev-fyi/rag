from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import dotenv
dotenv.load_dotenv()


from src.Llama_index_sandbox.data_ingestion_pdf.utils import return_driver
# Running a continuously open Selenium instance on Google Cloud Run is technically challenging and might not be the best approach for several reasons:
#
#     Statelessness and Timeouts: Cloud Run is designed for stateless, request-driven containers. Each container instance is expected to handle incoming requests, perform a task, and then quickly respond. Cloud Run instances have request timeouts (up to 60 minutes), after which the instance will be shut down if it hasn't completed its task.
#
#     Persistent State: A continuously open Selenium session requires a persistent state, which is contrary to the stateless nature of Cloud Run services.
#
#     Resource Consumption and Cost: Keeping a browser session open continuously can be resource-intensive. Cloud Run charges are based on the resources allocated and the time your container instances are running, so this might lead to higher costs.
#
#     Concurrency and Scaling: Cloud Run automatically scales the number of container instances based on the number of incoming requests. This behavior doesn't align well with the concept of maintaining a persistent Selenium session.
#
# A better approach might be to use Google Compute Engine (GCE), which provides more control over the environment and is better suited for long-running processes like an open Selenium session. Here’s how you can proceed:
#
#     Use GCE for Selenium Automation:
#         Set up a virtual machine on GCE.
#         Install the necessary software (like Chrome, ChromeDriver, and your Selenium script).
#         Run your script on this VM, which can handle the authentication and posting replies.
#
#     Cloud Run for Stateless Tasks:
#         Use Cloud Run for handling stateless, short-lived tasks.
#         When a task requiring browser interaction is needed, send a request to the GCE instance (possibly through a secure API).
#
#     Alternative - Dedicated Posting Service:
#         As you mentioned, consider having another process that solely handles posting. This service can run on a VM and manage the authenticated Selenium session.
#         Use message queues or a similar mechanism to send posting tasks to this dedicated service from your Cloud Run instances.


class TwitterSeleniumBot:
    def __init__(self, username, password):
        self.driver = return_driver()
        self.username = username
        self.password = password
        self.login_to_twitter()

    def login_to_twitter(self):
        login_url = "https://twitter.com/login"
        self.driver.get(login_url)
        time.sleep(2)

        # Input username
        username_field = self.driver.find_element("css selector", ".r-30o5oe")
        username_field.send_keys(self.username)
        time.sleep(5)

        # Click on "Next"
        next_button = self.driver.find_element("css selector", "div.css-175oi2r:nth-child(6) > div:nth-child(1)")
        next_button.click()
        time.sleep(5)

        # Input username
        # password_field = self.driver.find_element("css selector", ".r-homxoj")
        # password_field.send_keys('mevfyi')
        # time.sleep(5)

        # Input password
        password_field = self.driver.find_element("css selector", ".r-homxoj")
        password_field.send_keys(self.password)
        time.sleep(5)

        # Click on "Login"
        login_button = self.driver.find_element("css selector", "span.r-dnmrzs:nth-child(1) > span:nth-child(1)")
        login_button.click()
        time.sleep(3)

    def post_tweet_reply(self, tweet_id, reply_text):
        tweet_url = f"https://twitter.com/user/status/{tweet_id}"
        self.driver.get(tweet_url)
        time.sleep(3)

        # Locate the reply area (the specific method depends on Twitter's layout)
        # ...

        # Enter the reply text
        # ...

        # Submit the reply
        # ...

        # Optional: Close the driver or return it for further use
        # driver.quit()

    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    # Example usage

    twitter_username = os.environ.get('TWITTER_USERNAME')
    twitter_password = os.environ.get('TWITTER_PASSWORD')
    bot = TwitterSeleniumBot(twitter_username, twitter_password)

    example_tweet_id = "1234567890"
    example_reply_text = "This is a reply from Selenium bot."
    bot.post_tweet_reply(example_tweet_id, example_reply_text)

    # Close the bot when done
    bot.close()
