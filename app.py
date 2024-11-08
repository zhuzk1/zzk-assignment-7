from flask import Flask, render_template, request, session, url_for
import numpy as np
import matplotlib
from scipy.stats import t
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # 请替换为自己的密钥

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # 生成随机数据
    X = np.random.rand(N)
    Y = beta0 + beta1 * X + mu + np.random.normal(0, sigma2, N)

    # 拟合线性回归模型
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # 绘制散点图和回归线
    plt.scatter(X, Y, color="blue", label="Data")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.clf()

    # 模拟生成多个斜率和截距
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, sigma2, N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # 绘制斜率和截距的直方图
    plt.hist(slopes, bins=20, alpha=0.7, color="blue", label="Slopes")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.clf()

    # 计算比观察到的值更极端的比例
    slope_more_extreme = np.mean(np.abs(slopes) > np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) > np.abs(intercept))

    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 获取表单输入
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # 生成数据并绘制图形
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # 将数据存储在session中以便后续使用
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    session.clear()
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # 从session中获取数据
    print(session.get("N"))
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # 根据选择的参数和测试类型计算p值
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    if test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats) >= np.abs(observed_stat))
    elif test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    else:
        p_value = np.mean(simulated_stats <= observed_stat)

    fun_message = "Extremely significant result!" if p_value <= 0.0001 else None

    # 绘制假设检验的结果图
    plt.hist(simulated_stats, bins=20, color="lightgray")
    plt.axvline(observed_stat, color="red", linestyle="--", label="Observed Stat")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.clf()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        # 从 session 中获取数据
        N = int(session.get("N"))
        mu = float(session.get("mu"))
        sigma2 = float(session.get("sigma2"))
        beta0 = float(session.get("beta0"))
        beta1 = float(session.get("beta1"))
        S = int(session.get("S"))
        
        parameter = request.form.get("parameter")
        confidence_level = float(request.form.get("confidence_level"))

        # 获取模拟结果
        if parameter == "slope":
            estimates = np.array(session.get("slopes"))
            observed_stat = float(session.get("slope"))
            true_param = beta1
        else:
            estimates = np.array(session.get("intercepts"))
            observed_stat = float(session.get("intercept"))
            true_param = beta0

        # 使用 t 分布计算置信区间
        mean_estimate = np.mean(estimates)
        std_error = np.std(estimates, ddof=1) / np.sqrt(N)  # 使用样本标准差计算标准误差

        # 选择 t 分布的临界值
        alpha = 1 - confidence_level / 100
        df = N - 1  # 自由度
        t_value = t.ppf(1 - alpha / 2, df)

        # 计算置信区间的下限和上限
        ci_lower = mean_estimate - (t_value * std_error)
        ci_upper = mean_estimate + (t_value * std_error)
        includes_true = ci_lower <= true_param <= ci_upper

        # 绘制图形
        plt.figure(figsize=(8, 6))
        plt.scatter(estimates, [0] * len(estimates), color="gray", alpha=0.6, label="Simulated Estimates")
        plt.axvline(true_param, color="green", linestyle="--", linewidth=2.5, label="True " + parameter.capitalize())
        plt.plot([ci_lower, ci_upper], [0, 0], color="blue", linewidth=5, label=f"{confidence_level}% Confidence Interval")
        plt.plot(mean_estimate, 0, 'o', color="blue", markersize=10, label="Mean Estimate")

        plt.xlabel(f"{parameter.capitalize()} Estimate")
        plt.yticks([])  # 移除 y 轴刻度
        plt.title(f"{confidence_level}% Confidence Interval for {parameter.capitalize()} (Mean Estimate)")
        plt.legend(loc="upper right", frameon=True, framealpha=1, edgecolor="black")

        # 保存图像
        plot4_path = "static/plot4.png"
        plt.savefig(plot4_path, bbox_inches='tight')
        plt.clf()

        # 渲染模板并返回结果
        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot4=plot4_path,
            parameter=parameter,
            confidence_level=confidence_level,
            mean_estimate=mean_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            includes_true=includes_true,
            observed_stat=observed_stat,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    except Exception as e:
        print("An error occurred:", e)
        return "An error occurred while calculating the confidence interval."


if __name__ == "__main__":
    app.run(debug=True)
