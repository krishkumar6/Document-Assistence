"""Generate a small fictional PDF corpus with known facts on known pages.

Every section is forced onto its own page, so eval/questions.json can name the
exact page each answer lives on. Numbers like "30 days", "4 hours" and
"quarterly" deliberately recur across documents as retrieval distractors.

    python scripts/make_sample_pdfs.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "pdfs"

DOCS = {
    "northwind_drone_ops_manual.pdf": (
        "Northwind Aerial - Drone Operations Manual",
        [
            ("1. Fleet Overview", [
                "Northwind Aerial operates a mixed fleet from its main hangar in Tacoma. The survey fleet consists of 42 NW-7 quadcopters, which handle roof inspections, crop mapping and short-range photography. Heavy-lift work is done by 6 NW-12 hexacopters, which carry sensor pods and small cargo loads up to the regulatory limit.",
                "The NW-7 has a maximum takeoff weight of 2.4 kg including battery and camera gimbal. The NW-12 has a maximum takeoff weight of 11.8 kg and must only be flown by pilots holding the advanced endorsement. Each aircraft carries a serial plate under the battery bay, and that serial number is the key used in the maintenance log and in every flight record.",
                "Aircraft are assigned to crews at the start of each week by the fleet coordinator. A crew may not swap aircraft with another crew without updating the assignment board, because firmware versions and calibration dates are tracked per airframe. Aircraft that are grounded for maintenance are tagged with a red streamer on the left front arm and moved to the quarantine shelf.",
                "New pilots spend their first ten flights on the NW-7 under supervision before they are cleared for solo work. Supervision flights are logged with both the trainee and the supervising pilot named on the record.",
            ]),
            ("2. Pre-flight Checks", [
                "Before every launch the pilot completes the pre-flight checklist in the crew app. The flight battery must read at least 95 percent charge before takeoff; a battery below that level is swapped, not topped up in the field. Propellers are inspected for chips, cracks and warping, and any damaged propeller is replaced as a full set on that motor pair.",
                "Wind limits depend on the airframe. The NW-7 must not be launched when sustained wind exceeds 11 metres per second. The heavier NW-12 may be launched in sustained wind up to 15 metres per second, but gusts above that figure require the flight to be postponed. Pilots record the measured wind speed from the handheld anemometer, not the forecast.",
                "The compass must be calibrated every 30 days, and additionally whenever the aircraft is moved more than 500 km from its last calibration site. GPS lock with at least 12 satellites is required before arming the motors. The return-to-home altitude is set 20 metres above the tallest obstacle in the operating area.",
                "The visual observer confirms the launch zone is clear of people for a radius of 10 metres. If any checklist item fails, the flight is scrubbed and the failure is recorded with a photo in the crew app.",
            ]),
            ("3. Battery Management", [
                "Lithium polymer batteries are the most common cause of drone incidents, so they are handled under strict rules. Batteries that will not be used within 48 hours are brought to storage charge, which is 3.85 volts per cell. The smart chargers in the battery room do this automatically when set to storage mode.",
                "A battery is retired after 300 charge cycles, or immediately if it shows any swelling, puncture or a cell imbalance greater than 0.1 volts. Retired batteries are discharged fully in the salt-water bucket and sent to the certified recycler; they are never thrown in general waste.",
                "Charging takes place only in the battery room, where the temperature must stay between 10 and 30 degrees Celsius. Every battery charges inside a fireproof bag, and a charger is never left running while the room is unattended. A class D extinguisher and a sand bucket sit by the door.",
                "Each battery has a numbered label, and its cycle count is logged in the crew app at the end of every flying day. Batteries are rotated so that cycle counts across the pack stay roughly even.",
            ]),
            ("4. Incident Reporting", [
                "Any incident, including a crash, a flyaway, a near miss with a manned aircraft, or injury to any person, must be reported to the safety officer within 2 hours of the event. The report is filed on form NW-IR-3 in the crew app, with photos of the scene and the aircraft.",
                "In a flyaway, the pilot first attempts return-to-home, then switches to attitude mode and tries to regain manual control. If control cannot be regained, the pilot notes the last known position and heading and immediately notifies local air traffic control if the aircraft may enter controlled airspace.",
                "Flight logs and telemetry for any aircraft involved in an incident are exported the same day and retained for 18 months. The aircraft itself is grounded until the safety officer signs off the investigation, and it may not be repaired before photographs of the damage are taken.",
                "The safety officer reviews all incident reports at the monthly safety meeting, and lessons learned are added to this manual in the next revision.",
            ]),
        ],
    ),
    "helios_security_policy.pdf": (
        "Helios Cloud - Information Security Policy",
        [
            ("1. Scope and Roles", [
                "This policy applies to every employee, contractor and service account at Helios Cloud, and to every system that stores or processes customer data. The Chief Information Security Officer owns the policy and is accountable for its enforcement across engineering, support and finance.",
                "The policy is reviewed once a year, every March, by the security council. Changes approved at the March review take effect on the first day of the following quarter. Out-of-cycle changes may be made after a major incident, with sign-off from the CISO and the head of legal.",
                "Each team nominates a security champion who attends the monthly champions forum and acts as the first point of contact for security questions within the team. Security champions receive additional training and are expected to review designs that touch customer data.",
                "Exceptions to this policy must be requested in writing, approved by the CISO, and recorded in the exceptions register with an expiry date no more than 6 months away.",
            ]),
            ("2. Access Control", [
                "Multi-factor authentication is required for every account that can reach production systems, the source code host, or the customer support console. Hardware security keys are required for administrators; other staff may use an authenticator app.",
                "Passwords must be at least 14 characters long and must not appear in the breached-password list checked at login. Passwords are not rotated on a schedule, but must be changed immediately if compromise is suspected.",
                "Access reviews are performed quarterly. Each manager confirms that every member of their team still needs each role they hold, and unconfirmed roles are removed automatically at the end of the review window.",
                "When an employee leaves, all of their access must be revoked within 4 hours of their departure being recorded in the HR system. Shared credentials the person knew, such as break-glass accounts, are rotated within the same window.",
            ]),
            ("3. Data Classification and Protection", [
                "Helios classifies data into four levels: Public, Internal, Confidential and Restricted. Customer content and credentials are always Restricted. Financial records and employee data are Confidential. The data owner assigns the level and records it in the data catalogue.",
                "Restricted data must be encrypted at rest with AES-256 and in transit with TLS 1.2 or newer. Restricted data may not be copied to laptops, spreadsheets or chat tools under any circumstances.",
                "Backups of production databases are taken every 6 hours and retained for 35 days. Backups are encrypted with keys held in a separate account, and a restore test is performed every quarter to confirm the backups are usable.",
                "All company laptops use full-disk encryption and lock after 5 minutes of inactivity. Lost or stolen devices must be reported to IT the same day so they can be wiped remotely.",
            ]),
            ("4. Incident Response", [
                "Security incidents are graded Sev1 to Sev4. For a Sev1 incident, the on-call security engineer must be paged and must acknowledge within 15 minutes. The incident commander opens a dedicated channel and a timeline document immediately.",
                "If customer data is confirmed to have been exposed, affected customers are notified within 72 hours of confirmation. Regulators are notified where required by law, in coordination with the legal team.",
                "Evidence such as logs, disk images and access records is preserved before any remediation that might destroy it. Systems are isolated rather than powered off, so that memory can still be captured.",
                "A blameless postmortem is written within 5 business days of the incident being resolved. Action items from the postmortem are tracked in the security backlog and reviewed at the monthly security council meeting.",
            ]),
        ],
    ),
    "orbit_cafe_franchise_handbook.pdf": (
        "Orbit Cafe - Franchise Handbook",
        [
            ("1. The Brand and Franchise Fees", [
                "Orbit Cafe is a specialty coffee brand with a space-age design theme, known for its single-origin espresso and its signature Nebula cold brew. Franchisees operate under a ten-year agreement with an option to renew for a further ten years.",
                "The initial franchise fee is 45,000 US dollars, payable when the franchise agreement is signed. This fee covers brand licensing, the opening support team and access to the operations manual.",
                "Franchisees pay an ongoing royalty of 6 percent of gross sales, reported and paid monthly. In addition, 2 percent of gross sales is contributed to the national marketing fund, which pays for national campaigns, the mobile app and seasonal menu launches.",
                "Franchisees may not sell products that are not on the approved menu without written approval from the franchise support office. Local specials are permitted if they use approved ingredients.",
            ]),
            ("2. Opening a Store", [
                "Every proposed location must be approved by the real estate team before a lease is signed. Site approval takes up to 60 days and includes a footfall study and a review of nearby competitors. The minimum store size is 1,200 square feet of customer-facing space.",
                "Before opening, the franchisee and their store manager complete 3 weeks of training at the Orbit Academy in Denver. Training covers espresso preparation, the point-of-sale system, food safety and local store marketing.",
                "The opening support team spends the first 10 trading days on site. The store fit-out must use the approved design package, including the dome ceiling lighting and the star-map feature wall.",
                "A soft opening for friends and neighbours is recommended during the week before the public launch, so that the team can practise at full volume before real customers arrive.",
            ]),
            ("3. Food Safety", [
                "Refrigerators and milk fridges must hold a temperature of 4 degrees Celsius or below, checked and logged at opening, midday and closing. Any fridge reading above that limit for more than one check is reported to the store manager and the contents are assessed.",
                "Milk that has been left at room temperature for 4 hours or longer must be discarded. Opened milk cartons are labelled with the time they were opened.",
                "Staff wash their hands at least every 30 minutes during a shift, and always after handling cash, taking out waste or touching their face. Gloves are worn when handling ready-to-eat food.",
                "The espresso grinders and group heads are cleaned daily at close, and the espresso machine is descaled every week. Cleaning is signed off on the closing checklist by the shift lead.",
            ]),
            ("4. Quality Audits", [
                "Each store receives a mystery shopper visit twice per quarter. The shopper scores drink quality, speed of service, cleanliness and friendliness against the standard scorecard.",
                "The passing score for a quality audit is 85 out of 100. Scores are shared with the franchisee within a week of the visit, together with photos and notes from the shopper.",
                "A store that fails two consecutive audits must submit a remediation plan within 30 days. The regional manager then visits the store to agree the plan and schedules a follow-up audit.",
                "Stores that score 95 or above on every audit in a calendar year receive the Orbit Gold award and a reduction in their marketing fund contribution for the following year.",
            ]),
        ],
    ),
}


def build(path: Path, title: str, sections: list) -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (heading, paragraphs) in enumerate(sections):
        if i:
            story.append(PageBreak())
        else:
            story.append(Paragraph(title, styles["Title"]))
        story.append(Paragraph(heading, styles["Heading2"]))
        for p in paragraphs:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (title, sections) in DOCS.items():
        build(OUT / name, title, sections)
        print("wrote", OUT / name)
