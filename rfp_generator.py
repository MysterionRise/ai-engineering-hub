from faker import Faker

from main import do_action_as_chatgpt

fake = Faker()


def generate_company():
    return {
        "name": fake.company(),
        "address": fake.address(),
        "contact": fake.phone_number(),
        "email": fake.company_email(),
    }


def generate_fake_description(idea: str):
    return do_action_as_chatgpt(
        "You're generator of fake IT RFPs",
        f"You need to generate description based on specific ideaIdea: {idea}",
    )


def generate_fake_requirements(idea: str):
    return do_action_as_chatgpt(
        "You're generator of fake IT user stories for the following idea of system with up to 20 items",
        f"Be specific, provide user stories that are ready to be implemented, and follow INVEST principlesIdea: {idea}",
    )


def generate_fake_idea():
    return do_action_as_chatgpt(
        "You're generator of fake IT RFPs",
        "You need to generate main idea of that RFP in 1 catchy sentence",
    )


def generate_rfp(num_companies=5):
    idea = generate_fake_idea()
    rfp = {
        "title": f"RFP for {idea}",
        "description": generate_fake_description(idea),
        "requirements": generate_fake_requirements(idea).split("\n"),
        "proposal_deadline": fake.date_between(start_date="today", end_date="+3m"),
        "companies": [generate_company() for _ in range(num_companies)],
    }
    return rfp


def display_rfp(rfp):
    print(f"Title: {rfp['title']}\n")
    print(f"Description: {rfp['description']}\n")
    print("Requirements:")
    for requirement in rfp["requirements"]:
        print(f"- {requirement}")
    print(f"\nProposal Deadline: {rfp['proposal_deadline']}\n")
    print("Invited Companies:")
    for company in rfp["companies"]:
        print(f"\n{company['name']}")
        print(company["address"].replace("\n", ", "))
        print(f"Contact: {company['contact']}")
        print(f"Email: {company['email']}")


def save_rfp_to_file(rfp, filename="rfp_output.txt"):
    with open(filename, "w") as file:
        file.write(f"Title: {rfp['title']}\n\n")
        file.write(f"Description: {rfp['description']}\n\n")
        file.write("Requirements:\n")
        for requirement in rfp["requirements"]:
            file.write(f"- {requirement}\n")
        file.write(f"\nProposal Deadline: {rfp['proposal_deadline']}\n\n")
        file.write("Invited Companies:\n")
        for company in rfp["companies"]:
            file.write(f"\n{company['name']}\n")
            file.write(company["address"].replace("\n", ", ") + "\n")
            file.write(f"Contact: {company['contact']}\n")
            file.write(f"Email: {company['email']}\n")


if __name__ == "__main__":
    for _ in range(0, 1):
        idea = generate_fake_idea()
        user_stories = generate_fake_requirements(idea)
        print(user_stories)
