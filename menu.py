"""Shop data. Edit this file to change the shop, menu and promotions."""

RESTAURANT = {
    "name": "หมี่ไก่ฉีก เเดนเนรมิต",
    "hours": "เปิดทุกวัน 9 โมงเช้าถึง 2 ทุ่ม",
    "address": "ซอยสุขุมวิท 50 ใกล้ BTS อ่อนนุช",
}
PICKUP_MINUTES = 15  # orders are pickup only

# name -> price in baht ("special" = พิเศษ size)
MENU = {
    "หมี่ไก่ฉีก": {"price": 69, "special": 79},
    "หมี่คลุกซอสโบราณ": {"price": 69, "special": 79},
    "น้ำพริกกากหมู": {"price": 19},
}

# one promotion per bill, the biggest discount wins. weekday: 0 = Monday … 6 = Sunday
PROMOTIONS = [
    {"title": "สั่งครบ 300 บาท ลด 30 บาท", "min_total": 300, "discount": 30},
    {"title": "ทุกวันพุธ สั่งครบ 150 บาท ลด 10 เปอร์เซ็นต์", "min_total": 150, "percent": 10, "weekday": 2},
]
