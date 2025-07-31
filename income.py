import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QGridLayout,
    QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt


class TaxCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("收入计算器")
        self.resize(1000, 600)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # 左侧滚动输入区
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(20)

        # 基本信息
        base_group = QGroupBox("基本信息")
        base_layout = QGridLayout()
        base_layout.setHorizontalSpacing(15)
        base_layout.setVerticalSpacing(8)
        self.entry_salary = self.add_label_entry(base_layout, "月工资（元）", 0, "3000")
        self.entry_first_month_salary = self.add_label_entry(base_layout, "首月工资（元）", 1, "3000")
        self.entry_threshold = self.add_label_entry(base_layout, "起征点（元）", 2, "5000")
        self.entry_start_month = self.add_label_entry(base_layout, "起始月份（1-12）", 3, "1")
        self.entry_months = self.add_label_entry(base_layout, "计算月份数", 4, "12")
        self.entry_signon = self.add_label_entry(base_layout, "签字费（元）", 5, "0")
        self.entry_signon_month = self.add_label_entry(base_layout, "签字费月份", 6, "-1")
        base_group.setLayout(base_layout)

        # 五险一金比例
        rate_group = QGroupBox("五险一金缴费比例（%）【个人缴纳】")
        rate_layout = QGridLayout()
        rate_layout.setHorizontalSpacing(15)
        rate_layout.setVerticalSpacing(8)
        self.entry_pension = self.add_label_entry(rate_layout, "养老保险", 0, "8")
        self.entry_medical = self.add_label_entry(rate_layout, "医疗保险", 1, "2")
        self.entry_unemployment = self.add_label_entry(rate_layout, "失业保险", 2, "0.2")
        self.entry_fund = self.add_label_entry(rate_layout, "住房公积金", 3, "10")
        rate_group.setLayout(rate_layout)

        # 缴费基数上下限
        limit_group = QGroupBox("缴费基数上下限（元）")
        limit_layout = QGridLayout()
        limit_layout.setHorizontalSpacing(15)
        limit_layout.setVerticalSpacing(8)
        self.entry_pension_lower = self.add_label_entry(limit_layout, "养老下限", 0, "4492")
        self.entry_pension_upper = self.add_label_entry(limit_layout, "养老上限", 1, "27501")
        self.entry_medical_lower = self.add_label_entry(limit_layout, "医疗下限", 2, "6733")
        self.entry_medical_upper = self.add_label_entry(limit_layout, "医疗上限", 3, "33666")
        self.entry_unemp_lower = self.add_label_entry(limit_layout, "失业下限", 4, "2520")
        self.entry_unemp_upper = self.add_label_entry(limit_layout, "失业上限", 5, "44265")
        self.entry_fund_lower = self.add_label_entry(limit_layout, "公积金下限", 6, "2520")
        self.entry_fund_upper = self.add_label_entry(limit_layout, "公积金上限", 7, "44265")
        limit_group.setLayout(limit_layout)

        # 专项附加扣除
        extra_group = QGroupBox("专项附加扣除（元/月）")
        extra_layout = QGridLayout()
        extra_layout.setHorizontalSpacing(15)
        extra_layout.setVerticalSpacing(8)
        self.entry_edu = self.add_label_entry(extra_layout, "子女教育", 0, "0")
        self.entry_infant = self.add_label_entry(extra_layout, "婴幼儿照护", 1, "0")
        self.entry_loan = self.add_label_entry(extra_layout, "住房贷款利息", 2, "0")
        self.entry_rent = self.add_label_entry(extra_layout, "住房租金", 3, "1500")
        self.entry_elder = self.add_label_entry(extra_layout, "赡养老人", 4, "0")
        self.entry_illness = self.add_label_entry(extra_layout, "大病医疗（年）", 5, "0")
        self.entry_education_continue = self.add_label_entry(extra_layout, "继续教育（年）", 6, "0")
        extra_group.setLayout(extra_layout)

        # 添加所有分组到输入布局
        input_layout.addWidget(base_group)
        input_layout.addWidget(rate_group)
        input_layout.addWidget(limit_group)
        input_layout.addWidget(extra_group)

        # 计算按钮
        self.calc_button = QPushButton("计算")
        self.calc_button.setFixedHeight(40)
        input_layout.addWidget(self.calc_button)
        input_layout.addStretch(1)

        scroll_area.setWidget(input_container)
        main_layout.addWidget(scroll_area, 3)

        # 右侧结果显示
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFontFamily("Courier New")
        self.output_text.setStyleSheet("QTextEdit { padding: 10px; }")
        main_layout.addWidget(self.output_text, 5)

        self.calc_button.clicked.connect(self.calculate_net_salary)

    @staticmethod
    def add_label_entry(layout, label_text, row, default_value=""):
        label = QLabel(label_text)
        entry = QLineEdit()
        entry.setText(default_value)
        layout.addWidget(label, row, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(entry, row, 1)
        return entry

    @staticmethod
    def adjust_base(base, lower, upper):
        return min(max(base, lower), upper)

    @staticmethod
    def calculate_tax(taxable):
        brackets = [
            (36000, 0.03, 0),
            (144000, 0.10, 2520),
            (300000, 0.20, 16920),
            (420000, 0.25, 31920),
            (660000, 0.30, 52920),
            (960000, 0.35, 85920),
            (float('inf'), 0.45, 181920)
        ]

        tax = 0
        if taxable > 0:
            for limit, rate, deduction in brackets:
                if taxable <= limit:
                    tax = taxable * rate - deduction
                    break
        return tax

    def calculate_net_salary(self):
        try:
            salary = float(self.entry_salary.text())
            first_month_salary = float(self.entry_first_month_salary.text())
            pension_rate = float(self.entry_pension.text()) / 100
            medical_rate = float(self.entry_medical.text()) / 100
            unemployment_rate = float(self.entry_unemployment.text()) / 100
            fund_rate = float(self.entry_fund.text()) / 100
            threshold = float(self.entry_threshold.text())
            months = int(self.entry_months.text())
            start_month = int(self.entry_start_month.text())

            pension_lower = float(self.entry_pension_lower.text())
            pension_upper = float(self.entry_pension_upper.text())
            medical_lower = float(self.entry_medical_lower.text())
            medical_upper = float(self.entry_medical_upper.text())
            unemp_lower = float(self.entry_unemp_lower.text())
            unemp_upper = float(self.entry_unemp_upper.text())
            fund_lower = float(self.entry_fund_lower.text())
            fund_upper = float(self.entry_fund_upper.text())

            signon = float(self.entry_signon.text())
            signon_month = int(self.entry_signon_month.text())

            extra_deduction = (
                float(self.entry_edu.text()) +
                float(self.entry_infant.text()) +
                float(self.entry_loan.text()) +
                float(self.entry_rent.text()) +
                float(self.entry_elder.text())
            )

            illness_deduction = float(self.entry_illness.text())
            edu_continue_deduction = float(self.entry_education_continue.text())

            total_paid_tax = 0
            total_net = 0
            total_tax = 0
            total_pension_personal = 0
            total_medical_personal = 0
            total_unemp_personal = 0
            total_fund_personal = 0
            cumulative_income = 0
            cumulative_deduction = 0

            self.output_text.clear()
            header = f"{'月份':<6}{'到手工资':>8}{'个税':>9}{'医保个账':>11}{'养老金个账':>9}{'公积金账户':>9}"
            self.output_text.append(header)
            self.output_text.append("=" * 70)

            for i in range(months):
                month = start_month + i
                display_month = (month - 1) % 12 + 1
                month_name = f"{display_month}月"

                current_salary = first_month_salary if i == 0 else salary
                if month == signon_month:
                    current_salary += signon

                pension_base = self.adjust_base(salary, pension_lower, pension_upper)
                medical_base = self.adjust_base(salary, medical_lower, medical_upper)
                unemp_base = self.adjust_base(salary, unemp_lower, unemp_upper)
                fund_base = self.adjust_base(salary, fund_lower, fund_upper)

                pension = pension_base * pension_rate
                medical = medical_base * medical_rate
                unemp = unemp_base * unemployment_rate
                fund = fund_base * fund_rate
                insurance = pension + medical + unemp

                total_pension_personal += pension
                total_medical_personal += medical
                total_unemp_personal += unemp
                total_fund_personal += fund

                cumulative_income += current_salary
                cumulative_deduction += insurance + fund + threshold

                taxable_income = cumulative_income - cumulative_deduction
                tax_all = self.calculate_tax(taxable_income)
                tax_month = max(0, tax_all - total_paid_tax)
                total_paid_tax = tax_all

                net_income = current_salary - insurance - fund - tax_month
                total_net += net_income
                total_tax += tax_month

                line = f"{month_name:>3}{net_income:15.2f}{tax_month:11.2f}{medical:12.2f}{pension:12.2f}{fund * 2:12.2f}"
                self.output_text.append(line)

            self.output_text.append("=" * 70)
            summary = f"{'合计':<6}{total_net:12.2f}{total_tax:11.2f}{total_medical_personal:12.2f}{total_pension_personal:12.2f}{total_fund_personal * 2:12.2f}"
            self.output_text.append(summary)

            final_total_income = cumulative_income
            final_total_deduction = (
                threshold * 12 +
                extra_deduction * months +
                total_pension_personal +
                total_medical_personal +
                total_fund_personal +
                total_unemp_personal +
                illness_deduction +
                edu_continue_deduction
            )
            final_taxable_income = final_total_income - final_total_deduction
            final_tax = self.calculate_tax(final_taxable_income)
            refund = total_tax - final_tax

            self.output_text.append(f"\n年度退税: {refund:,.2f}")

        except ValueError:
            self.output_text.clear()
            self.output_text.append("⚠️ 请输入有效数字！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(10)  # 调大字体大小，例如设置为12
    app.setFont(font)

    window = TaxCalculator()
    window.show()
    sys.exit(app.exec())

