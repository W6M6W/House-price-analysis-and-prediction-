const translations = {
    zh: {
        pageTitle: "房价智能评估系统",
        headerTitle: "房产价值评估工具",
        headerSubtitle: "基于大数据算法精准估算您的房产价值",
        areaLabel: "房屋面积（平方米）",
        directionLabel: "朝向",
        southOption: "南向",
        northOption: "北向",
        eastOption: "东向",
        westOption: "西向",
        floorLevelLabel: "所在楼层",
        highFloor: "高楼层",
        mediumFloor: "中楼层",
        lowFloor: "低楼层",
        totalFloorsLabel: "总楼层数",
        yearLabel: "建成年代",
        addressLabel: "详细地址",
        addressPlaceholder: "例如：衡阳市华新开发区XX路XX小区",
        submitBtn: "立即评估",
        resultTitle: "预估结果",
        factorsTitle: "影响因素",
        errorInvalidInput: "请输入有效的数据",
        errorApiFail: "评估失败，请检查输入后重试",
        loadingText: "评估时间较长，需要一到三分钟左右，请耐心等待",
        regionImpact: "本区域房价",
        floorImpact: "楼层影响",
        floorHeight: "层高影响",
        priceComparison: "均价对比",
        regionalAverage: "区域均价",
        cityAverage: "全市均价",
        closeResults: "X",
        requestTimeout: "请求超时，请稍后再试"
    },
    en: {
        pageTitle: "House Price Estimation System",
        headerTitle: "Property Value Estimation Tool",
        headerSubtitle: "Accurately estimate your property value using big data algorithms",
        areaLabel: "Area (sqm)",
        directionLabel: "Direction",
        southOption: "South",
        northOption: "North",
        eastOption: "East",
        westOption: "West",
        floorLevelLabel: "Floor Level",
        highFloor: "High",
        mediumFloor: "Middle",
        lowFloor: "Low",
        totalFloorsLabel: "Total Floors",
        yearLabel: "Year Built",
        addressLabel: "Address",
        addressPlaceholder: "e.g., XX Community, XX Road, Huaxin Development Zone, Hengyang",
        submitBtn: "Evaluate Now",
        resultTitle: "Estimated Result",
        factorsTitle: "Influencing Factors",
        errorInvalidInput: "Please enter valid data",
        errorApiFail: "Estimation failed, please check your input and try again",
        loadingText: "Evaluation may take 1-3 minutes, please wait patiently",
        regionImpact: "Regional Impact",
        floorImpact: "Floor Impact",
        floorHeight: "Floor Height",
        priceComparison: "Price Comparison", 
        regionalAverage: "Regional Avg",
        cityAverage: "City Avg",
        closeResults: "X",
        requestTimeout: "Request timeout, please try again later"
    }
};

let currentLang = 'zh';
const loadingModal = document.getElementById('loadingModal');

// 语言切换功能
document.getElementById('langSwitch').addEventListener('click', function() {
    currentLang = currentLang === 'zh'? 'en' : 'zh';
    updateLanguage();
    this.textContent = currentLang === 'zh'? "English" : "中文";
});

// 更新页面文本
function updateLanguage() {
    // 更新常规元素
    document.querySelectorAll('[data-i18n]').forEach(elem => {
        const key = elem.getAttribute('data-i18n');
        if (translations[currentLang]?.[key]) {
            elem.textContent = translations[currentLang][key];
        }
    });

    // 更新 placeholder
    document.querySelectorAll('[data-i18n-placeholder]').forEach(elem => {
        const key = elem.getAttribute('data-i18n-placeholder');
        if (translations[currentLang]?.[key]) {
            elem.placeholder = translations[currentLang][key];
        }
    });

    // 更新页面标题与关闭按钮文本
    document.title = translations[currentLang].pageTitle;
    document.getElementById('closeResults').textContent = translations[currentLang].closeResults;

    // 更新卡片标题
    document.querySelectorAll('.data-card h3[data-i18n]').forEach(elem => {
        const key = elem.getAttribute('data-i18n');
        elem.textContent = translations[currentLang][key];
    });
}

// 表单提交处理
document.getElementById('submitBtn').addEventListener('click', async function() {
    let timeoutId;
    const TIMEOUT_DURATION = 300000; // 5分钟

    const formData = {
        area: parseFloat(document.getElementById('area').value),
        direction: document.getElementById('direction').value,
        floorLevel: document.querySelector('input[name="floorLevel"]:checked').value,
        totalFloors: parseInt(document.getElementById('totalFloors').value),
        year: parseInt(document.getElementById('year').value),
        address: document.getElementById('address').value.trim()
    };

    // 输入验证
    if (!formData.area ||!formData.totalFloors ||!formData.year) {
        alert(translations[currentLang].errorInvalidInput);
        return;
    }

    try {
        // 显示加载弹窗
        loadingModal.style.display = 'flex';
        
        // 设置超时定时器
        timeoutId = setTimeout(() => {
            if (loadingModal.style.display === 'flex') {
                throw new Error(translations[currentLang].requestTimeout);
            }
        }, TIMEOUT_DURATION);

        const response = await fetch('http://localhost:5000/evaluate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(formData)
        });

        clearTimeout(timeoutId);

        if (!response.ok) throw new Error(translations[currentLang].errorApiFail);
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // 显示结果弹窗
        updateResult(data);
        
    } catch (error) {
        clearTimeout(timeoutId);
        console.error('Error:', error);
        alert(error.message);
        document.getElementById('resultContainer').style.display = 'none';
        document.getElementById('resultOverlay').style.display = 'none';
    } finally {
        loadingModal.style.display = 'none';
    }
});

// 结果处理函数：横向滚动效果 + 卡片美化设计
function updateResult(data) {
    const cardWrapper = document.getElementById('cardWrapper');
    const closeBtn = document.getElementById('closeResults');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    // 清空旧卡片
    while (cardWrapper.firstChild) {
        cardWrapper.removeChild(cardWrapper.firstChild);
    }

    // 生成卡片数据
    const cards = [
        { title: "预估结果:", value: data.estimate, unit: '元/每平方米', image: "卡片底图.gif" },
        { title: "朝向影响:", value: data.factors[0], unit: '%', image: "卡片底图1.png" },
        { title: "总楼层影响:", value: data.factors[1], unit: '%', image: "卡片底图2.png" },
        { title: "所在楼层影响:", value: data.factors[2], unit: '%', image: "卡片底图3.png" },
        { title: "所在区域均价:", value: data.factors[3], unit: '元/每平方米', image: "卡片底图4.png" },
        { title: "全市均价:", value: data.whole_aver, unit: '元/每平方米', image: "卡片底图5.png" }
    ];

    // 为每个卡片生成 DOM
    cards.forEach((cardData, index) => {
        const card = document.createElement('div');
        card.className = 'data-card';

        // 添加图片
        const img = document.createElement('img');
        img.className = 'card-image';
        img.src = cardData.image;
        card.appendChild(img);

        // 添加标题
        const titleDiv = document.createElement('div');
        titleDiv.className = 'card-title';
        titleDiv.textContent = cardData.title;
        card.appendChild(titleDiv);

        // 添加值
        const valueDiv = document.createElement('div');
        valueDiv.className = 'card-value';
        valueDiv.textContent = `${cardData.value} ${cardData.unit}`;
        card.appendChild(valueDiv);

        cardWrapper.appendChild(card);
    });

    // 显示结果与导航
    document.getElementById('resultOverlay').style.display = 'block';
    resultContainer.style.display = 'block';

    // 卡片切换逻辑
    let currentIndex = 0;

    prevBtn.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + cards.length) % cards.length;
        updateCardPosition();
    });

    nextBtn.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % cards.length;
        updateCardPosition();
    });

    closeBtn.addEventListener('click', () => {
        document.getElementById('resultOverlay').style.display = 'none';
        resultContainer.style.display = 'none';
    });

    function updateCardPosition() {
        // 移动卡片位置
        const cardWidth = cardWrapper.querySelector('.data-card').offsetWidth + 40; // 卡片宽度 + 间距
        cardWrapper.style.transform = `translateX(-${currentIndex * cardWidth}px)`;

        // 更新选中状态
        cardWrapper.querySelectorAll('.data-card').forEach((card, index) => {
            if (index === currentIndex) {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        });

        // 让选中的卡片居中
        const selectedCard = cardWrapper.children[currentIndex];
        selectedCard.scrollIntoView({
            behavior: 'smooth',
            inline: 'center' // 水平居中
        });
    }

    // 初始化位置
    updateCardPosition();
}


