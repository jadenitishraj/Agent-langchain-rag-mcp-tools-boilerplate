"""
Contact Tool - Returns the application owner's contact information

This tool is called when users ask for:
- Email address
- Phone number
- Contact information
- How to reach the owner/admin
- Support contact details
"""
from langchain_core.tools import tool
from typing import Literal

# Hardcoded contact information (replace with your actual details)
OWNER_EMAIL = "support@agentforge.dev"
OWNER_PHONE = "+1-555-123-4567"
OWNER_NAME = "AgentForge Support"

@tool
def get_contact_info(info_type: Literal["email", "phone", "all"] = "all") -> str:
    """
    Get the contact information of the application owner/support team.
    
    Use this tool when the user asks for:
    - Email address of the owner/admin/support
    - Phone number to contact
    - How to reach support or the team
    - Contact details or contact information
    - Ways to get in touch
    
    Args:
        info_type: Type of contact info to retrieve. 
                   "email" for email only, "phone" for phone only, "all" for both.
    
    Returns:
        The requested contact information.
    """
    if info_type == "email":
        return f"📧 Email: {OWNER_EMAIL}"
    elif info_type == "phone":
        return f"📱 Phone: {OWNER_PHONE}"
    else:
        return f"""
📞 Contact Information:
━━━━━━━━━━━━━━━━━━━━━━
👤 Name: {OWNER_NAME}
📧 Email: {OWNER_EMAIL}
📱 Phone: {OWNER_PHONE}

Feel free to reach out for any questions or support!
"""

# Alternative function for direct use
def contact_tool() -> dict:
    """Returns contact info as a dictionary."""
    return {
        "name": OWNER_NAME,
        "email": OWNER_EMAIL,
        "phone": OWNER_PHONE
    }

# List of all tools for easy import
ALL_TOOLS = [get_contact_info]
