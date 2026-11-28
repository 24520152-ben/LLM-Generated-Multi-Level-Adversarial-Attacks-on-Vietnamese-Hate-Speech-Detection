# LLM-Generated Multi-Level Adversarial Attacks on Vietnamese Hate Speech Detection

*Tấn công đối kháng đa cấp độ bằng LLM đối với phát hiện ngôn từ thù ghét tiếng Việt*

> Cập nhật 29/11/2026. Đã có baseline seed 42, attack pilot của bốn classifier và kết quả ba LLM judge trên Kaggle. Phiên bản hiện tại kiểm định bằng Qwen và SEA-LION độc lập, Gemma phân xử khi cần; human review và nâng cấp judge được hoãn sang phiên bản sau. Mọi kết quả hợp lệ được báo cáo là LLM-judged, chưa được human xác nhận.

## Abstract

Các chỉnh sửa nhỏ hoặc cách diễn đạt khác có thể khiến mô hình phát hiện ngôn từ thù ghét dự đoán sai dù nội dung vẫn giữ nguyên. Nghiên cứu này sử dụng **một LLM với ba chế độ sinh ứng viên** ở cấp ký tự, từ/cụm từ và câu, kế thừa hướng prompting đa cấp độ của PromptAttack. Các chế độ dùng chung beam search, mục tiêu gây lỗi và giao thức kiểm định; điều kiện phối hợp được đánh giá dưới cùng ngân sách truy vấn classifier và vị trí sinh; chi phí sinh LLM thực tế được đo riêng. Thí nghiệm sử dụng ViHSD với ba nhãn `CLEAN`, `OFFENSIVE` và `HATE`. Kết quả baseline ban đầu, seed 42, đạt test macro-F1 từ 61,83% đến 66,67%; pilot đã có kết quả lật nhãn tự động và kiểm định bằng ba LLM judge. Hai judge chấm độc lập, judge thứ ba phân xử bất đồng; human review tạm hoãn. Kết quả được diễn giải là LLM-judged validity. Nghiên cứu hướng tới cung cấp tập kiểm thử đối kháng được kiểm định và phân tích hiệu quả, mức tuân thủ từng chế độ, khả năng phối hợp và chuyển giao giữa mô hình.

## 1. Introduction

Phát hiện ngôn từ thù ghét trên mạng xã hội cần hoạt động ổn định khi người dùng thay đổi cách viết hoặc cách diễn đạt. Với tiếng Việt, từ nhiều âm tiết, ranh giới từ và hệ dấu tạo ra những yêu cầu riêng khi thích nghi attack văn bản. Một ứng viên trôi chảy hoặc có độ tương đồng ngữ nghĩa cao vẫn có thể làm đổi đối tượng bị công kích hoặc nhãn đúng.

Các phương pháp hiện có sử dụng nhiều cơ chế: TextBugger chỉnh sửa ký tự [2], TextFooler thay từ bằng tài nguyên từ vựng [3], BERT-Attack sinh thay thế theo ngữ cảnh [4], còn TREPAT khai thác bản viết lại từ mô hình ngôn ngữ [6]. PromptAttack [13] đã dùng LLM sinh attack ở cả ba cấp độ. Nghiên cứu này kế thừa cách điều khiển bằng prompt, giữ chung LLM và bộ tìm kiếm để khảo sát các chế độ chỉnh sửa trên Vietnamese hate speech detection. LLM sinh ứng viên được tách khỏi classifier đích, cho phép tấn công cả mô hình truyền thống và encoder.

Phiên bản hiện tại xét H1–H2 bằng LLM-judged validity; H3 được giữ cho hướng mở rộng sau:

- **H1:** Với cùng LLM sinh ứng viên và beam search, hiệu quả của ba chế độ phụ thuộc vào mô hình đích; không có chế độ chiếm ưu thế trong mọi điều kiện khảo sát.
- **H2:** Phối hợp các cấp độ tăng tỷ lệ attack thành công hợp lệ so với từng cấp độ riêng dưới cùng ngân sách truy vấn classifier, vị trí sinh và quy tắc kiểm định hợp lệ; lượt gọi LLM và thời gian tìm kiếm được báo cáo riêng.
- **H3 (tạm hoãn):** Kiểm định bởi con người làm thay đổi đáng kể ước lượng hiệu quả của ít nhất một điều kiện attack; khả năng đổi thứ hạng được phân tích bổ sung.

Đóng góp dự kiến gồm: (1) giao thức thích nghi và kiểm soát ba chế độ sinh attack bằng một LLM cho tiếng Việt; (2) tập kiểm thử đối kháng dẫn xuất từ ViHSD, có LLM annotation và truy vết; (3) phân tích thực nghiệm kiểm định các giả thuyết trên. Prompting đa cấp độ và beam search được kế thừa; giá trị nghiên cứu cần được xác lập từ chất lượng tài nguyên và kết quả thực nghiệm.

## 2. Related Work

**Character-level attacks.** TextBugger [2] tìm các chỉnh sửa ký tự và từ gây lỗi thông qua phản hồi mô hình. Các thao tác ký tự là tài liệu tham khảo để thiết kế hướng dẫn và kiểm tra đầu ra LLM trong chế độ ký tự.

**Word/span-level attacks.** TextFooler [3] sử dụng ứng viên gần nghĩa; BERT-Attack [4] dùng masked language model để đề xuất thay thế theo ngữ cảnh. FBA [5] kết hợp thao tác từ với tìm kiếm Metropolis–Hastings. Các hướng này cung cấp cơ sở đối chiếu với thay thế từ theo ngữ cảnh do LLM sinh.

**LLM-based multi-level attacks.** PromptAttack của Xu và cộng sự [13], công bố tại ICLR 2024, dùng prompt gồm đầu vào gốc, mục tiêu và hướng dẫn chỉnh sửa ở cấp ký tự, từ và câu, kết hợp fidelity filtering. TREPAT [6] sử dụng bản viết lại từ mô hình ngôn ngữ và beam search. Chúng tôi kế thừa các ý tưởng này để đánh giá một LLM sinh ứng viên cho classifier độc lập, với kiểm định ba nhãn ViHSD và đối chứng cho lợi ích phối hợp cấp độ.

CoPA [14], EMNLP 2025 Main, dùng Qwen2.5-72B-Instruct làm paraphraser chính để tấn công bộ phát hiện văn bản máy sinh. Đây là tiền lệ cho LLM sinh bản viết lại, nhưng khác nhiệm vụ phát hiện ngôn từ thù ghét. TREPAT thử Llama-3.2-1B/3B-Instruct, Gemma-2-2B/9B-it, OLMo-7B-Instruct và Llama-3.1-8B-Instruct [6]; các model nhỏ cung cấp tiền lệ triển khai local. Pilot trước dùng Qwen3-8B; lượt chạy lại trên validation chuyển sang Qwen3-14B 4-bit, cần đo lại chất lượng trên tiếng Việt.

**Evaluation and Vietnamese NLP.** TextAttack [7] phân tách mục tiêu, ràng buộc, biến đổi và tìm kiếm; Morris và cộng sự [8] cho thấy cần kiểm định tính hợp lệ của adversarial examples. Lent [9] đã đưa tiếng Việt vào thử nghiệm đa ngôn ngữ với TextFooler trên phân loại chủ đề. Đề tài tập trung vào ViHSD [1], ba cấp độ chỉnh sửa và yêu cầu giữ nhãn của tác vụ, thay vì giả định adversarial NLP tiếng Việt chưa có tiền lệ.

## 3. Method

### 3.1. Threat Model and Objective

Thiết lập chính là **black-box, score-based, untargeted evasion**: attacker biết câu gốc và nhãn đánh giá, được truy vấn điểm số đầu ra nhưng không dùng gradient hoặc trọng số classifier. Mỗi đầu vào đi qua toàn bộ pipeline tiền xử lý và tokenizer của mô hình đích.

**Rubric nhãn.** Theo guideline ViHSD [1, Bảng 1], `CLEAN` (0) không có nội dung công kích/quấy rối; `OFFENSIVE` (1) có nội dung xúc phạm hoặc chửi tục nhưng không công kích đối tượng cụ thể; `HATE` (2) công kích một cá nhân hoặc một nhóm người, gồm cả công kích hàm ý/nghĩa bóng không có từ tục rõ ràng. `HATE` không chỉ giới hạn ở nhóm được bảo vệ. Prompt dùng rubric chung trong `VIHSD_LABEL_RUBRIC` tại `constants.py`; tiếng lóng, viết tắt và cách viết né tránh được xét theo ngữ cảnh, không tự suy diễn đối tượng hay bổ sung ngữ cảnh thiếu. Nhãn gốc là điều kiện cần bảo toàn; bộ sinh không gán nhãn lại hoặc sửa nghĩa để ép khớp rubric. Rubric hướng dẫn sinh, chưa xác nhận tính hợp lệ của ứng viên.

Với câu gốc $x$, biến thể $x'$, nhãn nguồn $y$ và classifier $f$, một attack được bộ judge chấp nhận phải thỏa:

$$
f(x) = y,\qquad f(x') \ne y,\qquad V_{\mathrm{LLM}}(x,x',y) = 1.
$$

$V_{\mathrm{LLM}} = 1$ khi kết quả tổng hợp của ba judge là `valid=YES`, tức bộ judge đánh giá biến thể giữ nghĩa, giữ nhãn và các bước đúng chế độ. Đây là quyết định bằng LLM; human review không thuộc phiên bản hiện tại. Bộ lọc cấu trúc trong tìm kiếm chỉ kiểm tra định dạng và giới hạn chỉnh sửa, không tự xác nhận các điều kiện này.

Ngân sách $Q$ giới hạn truy vấn classifier. Ngân sách $G$ giới hạn vị trí ứng viên được yêu cầu sinh (tối đa 100 trong pilot) cho mỗi câu/chế độ/beam, tính trước khi lọc. LLM sinh đề xuất ứng viên; phản hồi classifier dẫn hướng tìm kiếm; ba LLM judge kiểm định sau tìm kiếm.

Gọi $s_c(x')$ là điểm đầu ra lớp $c$. Beam search ưu tiên ứng viên có margin lớn:

$$
g_f(x',y) = \max_{c \ne y} s_c(x') - s_y(x').
$$

Chỉ so sánh điểm trong cùng một classifier. SVM dùng decision score; các encoder dùng điểm đầu ra nhất quán với quy tắc dự đoán của chúng.

### 3.2. One LLM, Three Generation Modes

Với cùng checkpoint $\theta$, ba chế độ dùng chung LLM sinh ứng viên:

$$
G_{\ell}(z;x,y) = \mathrm{LLM}_{\theta}(P_{\mathrm{shared}},P_{\ell},x,y,z),\qquad \ell \in \{C,W,S\}.
$$

$z$ là câu đang mở rộng; $x$, $y$ là câu gốc và nhãn nguồn. $P_{\mathrm{shared}}$ quy định đầu ra, giữ nghĩa/nhãn và giới hạn chỉnh sửa; $P_{\ell}$ quy định cấp độ. Cung cấp câu gốc ở mỗi vòng để kiểm soát trôi nghĩa tích lũy.

| Chế độ | Hướng dẫn sinh ứng viên | Kiểm tra tuân thủ |
|---|---|---|
| **Ký tự ($G_C$)** | Tạo lỗi chèn, xóa, đảo hoặc thay ký tự; mỗi ứng viên khác câu hiện tại, chép nguyên phần không sửa | Chuẩn hóa NFC, Levenshtein trên grapheme cluster từ 1 đến 2 thao tác mỗi bước; đảo hai ký tự tính 2 thao tác; LLM judge kiểm tra đúng loại chỉnh sửa |
| **Từ/cụm từ ($G_W$)** | Thay một số từ/cụm từ bằng cách diễn đạt gần nghĩa trong ngữ cảnh; giữ cấu trúc phần còn lại | Không quá 2 vùng diff; mỗi vùng trước/sau không quá 4 khối chữ/số Unicode; cho phép mở rộng/rút gọn cụm từ; LLM judge đánh giá cấp độ |
| **Câu ($G_S$)** | Paraphrase hoặc thay cấu trúc cú pháp, bảo toàn nội dung quyết định nhãn | Chỉ kiểm tra nội dung chữ/số thực sự thay đổi; không dùng số vùng diff để xác nhận S; LLM judge đánh giá paraphrase/cú pháp |

Lượt chạy lại dùng checkpoint chính thức `Qwen/Qwen3-14B`, quantize NF4 4-bit với double quantization khi nạp, tính toán FP16 và tắt thinking [15]. File trọng số gốc cần tải khoảng 27,51 GiB, ngoài dung lượng môi trường Python và cache; không đồng nhất dung lượng đĩa với VRAM sau quantize. Số ứng viên mặc định là 5/lượt, một lựa chọn triển khai được đánh giá trên validation. Temperature 0.7, top-p 0.8 và top-k 20 theo khuyến nghị non-thinking của Qwen3-14B [15]. Model được chọn nhằm tăng dung lượng bộ sinh trong giới hạn bộ nhớ một T4; chưa có bằng chứng nó tối ưu cho ViHSD. Các lựa chọn này được kiểm tra trên validation rồi đóng băng checkpoint/revision, khung prompt, tham số giải mã và số ứng viên mỗi lượt gọi. Metadata chỉnh sửa được tính lại từ văn bản đầu ra. Bộ lọc cấu trúc không xác nhận giữ nghĩa/nhãn. LLM judge chấm gộp giữ nghĩa và tuân thủ cấp độ; nhãn được gán độc lập rồi code đối chiếu với nhãn nguồn. Trường hợp chưa rõ được giữ UNSURE, chưa chấp nhận làm thành công theo LLM.

**Hiệu chỉnh prompt C sau pilot.** Trong CSV chưa dùng cache, 219/222 đầu ra C bị loại bởi bộ lọc `mode` của ViSoBERT và 217/223 của XLM-R không thay đổi so với câu cha. Các số này đếm theo dòng, có lặp giữa các beam, không phải mẫu độc lập. Prompt cũ chỉ nói tối đa hai chỉnh sửa, chưa yêu cầu tối thiểu một; bản sửa làm rõ 1–2 thao tác trên toàn câu và cách đếm grapheme, yêu cầu ứng viên khác nhau, chép nguyên phần không sửa và không điền câu giữ nguyên khi thiếu biến thể. C thay đổi nội dung hướng dẫn; W/S chỉ được trình bày lại thành chuỗi nhiều dòng, giữ nguyên từ ngữ. Bộ lọc giữ nguyên.

**Cơ sở thiết kế.** PromptAttack [13, Mục 3.1/3.3 và Bảng 22] cung cấp hướng typo/few-shot, nhưng ghi nhận model nhỏ có thể sao chép định dạng ví dụ. Bản thích nghi dùng hai ví dụ tiếng Việt tự soạn, mỗi ví dụ là một đầu vào và JSON array năm biến thể, cùng định dạng đầu ra thực tế; số lượng này là lựa chọn pilot, chưa phải cấu hình few-shot tối ưu. Chèn/xóa/đảo/thay ký tự tham khảo TextBugger [2, Mục III-C], không triển khai lại bộ sinh heuristic của paper. ANTHRO [16], Findings ACL 2022, bổ sung cơ sở ưu tiên biến thể có cách đọc tự nhiên thay vì chỉ tăng lỗi classifier. Dấu tiếng Việt có thể gây nhập nhằng từ vựng [17], nên prompt chỉ cho sửa dấu cục bộ khi vẫn nhận ra cùng từ trong ngữ cảnh, đồng thời giữ phủ định, đối tượng và nhãn. Mười biến thể minh họa đều vượt bộ lọc C hiện tại ngay cả khi giới hạn một thao tác; điều này chỉ xác nhận cấu trúc. Cần đo lại tỷ lệ giữ nguyên/lặp, mức tuân thủ và chất lượng trên validation bằng Qwen3 thật cho cả C và MIX trước khi kết luận cải thiện.

Bộ lọc pilot loại câu trùng lặp, hỏng định dạng hoặc không đạt giới hạn hình thức. C giới hạn 2 thao tác Levenshtein trên grapheme cluster mỗi bước. W giới hạn 2 vùng diff, mỗi vùng tối đa 4 đơn vị chữ/số ở cả trước và sau; đơn vị được tách bằng regex `[\p{L}\p{N}]+`, không phải từ tiếng Việt đã tách nghĩa. Diff có thể là thay, chèn hoặc xóa để không loại nhầm mở rộng/rút gọn cụm từ. Không còn loại W/S chỉ vì khoảng cách ký tự nhỏ; các bộ lọc có thể chồng lấn. W vẫn có thể nhận thay đổi sai nghĩa hoặc cả bình luận ngắn nếu nằm trong giới hạn. S chỉ yêu cầu có nội dung chữ/số thay đổi, bỏ điều kiện ít nhất hai vùng diff; việc vượt bộ lọc S không chứng minh paraphrase hay đổi cú pháp. W/S loại thay đổi chỉ ở khoảng trắng, dấu câu hoặc chữ hoa/thường. Trong tìm kiếm chưa có bộ lọc ngữ nghĩa. Sau tìm kiếm, ba LLM judge đánh giá cấp độ, giữ nghĩa và giữ nhãn; không dùng riêng chính LLM sinh để tự chứng nhận. Đầu ra không đạt hoặc từ chối sinh được ghi nhận cùng lượt gọi sinh. Kết quả judge hiện chưa được hiệu chuẩn bằng human.

### 3.3. Unified Beam Search

Khởi tạo beam bằng câu gốc. Ở mỗi vòng, bộ sinh được phép tạo ứng viên từ các câu trong beam; hệ thống lọc, loại trùng, truy vấn classifier và giữ tối đa $b$ câu có margin tốt nhất để mở rộng. Mọi ứng viên đã chấm được lưu trong candidate pool, kể cả khi không được chọn vào beam tiếp theo; các câu gây lỗi được đưa sang kiểm định. Margin chỉ quyết định ưu tiên tìm kiếm, không quyết định biến thể nào được giữ trong tập test cuối cùng. Tìm kiếm dừng khi hết ngân sách truy vấn $Q$, hết vị trí sinh $G$ hoặc một vòng không có ứng viên mới được chấm. Không dừng ở lần đổi nhãn đầu tiên để có thể kiểm định các ứng viên gây lỗi tiếp theo.

Bốn điều kiện dùng cùng bộ tìm kiếm là **C**, **W**, **S** và **MIX**. MIX cho phép một chuỗi chỉnh sửa đi qua nhiều chế độ của cùng LLM; cả $Q$ và $G$ được dùng chung cho toàn bộ chuỗi và có cùng giới hạn như attack đơn cấp. Mỗi parent được mở rộng bằng cả ba chế độ, với thứ tự bắt đầu luân phiên theo câu và vòng, độc lập với Q; kiểm tra chế độ xét bước chỉnh sửa hiện tại. Chi phí LLM có thể khác nhau giữa các điều kiện và được báo cáo riêng.

Đánh giá câu gốc để xác định tập đủ điều kiện là bước chuẩn bị riêng, ngoài ngân sách attack. Mọi đánh giá classifier trong quá trình tìm kiếm đều được tính vào ngân sách; một batch chứa $n$ đầu vào tương ứng $n$ đánh giá. Riêng kiểm tra tuân thủ chế độ ký tự dùng số thao tác Levenshtein trên grapheme cluster của **bước hiện tại**; thí nghiệm không tính hay lưu khoảng cách giữa câu gốc và biến thể. Lịch sử các chế độ chỉnh sửa được lưu để truy vết chuỗi MIX.

## 4. Adversarial Test Set Construction

Dữ liệu nguồn là test set ViHSD [1], giữ ba nhãn theo hướng dẫn corpus. Pilot và phát triển phương pháp sử dụng validation. ID câu test, quy mô lấy mẫu, ngân sách và annotation guideline được khóa trước thí nghiệm chính. Nếu lấy mẫu, sử dụng quy tắc phân tầng theo nhãn, độc lập với kết quả attack. Phiên bản hiện tại giữ nhãn nguồn ViHSD và chưa kiểm định/gán lại nhãn câu gốc bằng human. Sai lệch nhãn nguồn là một hạn chế cần xét khi diễn giải kết quả judge.

**Kiểm định bằng ba LLM.** Chỉ đưa các biến thể đã làm classifier bị tấn công dự đoán khác nhãn nguồn sang kiểm định. Hai judge `Qwen/Qwen3.5-9B` và `aisingapore/Llama-SEA-LION-v3.5-8B-R` chấm độc lập; chỉ gọi `google/gemma-4-12B-it` khi nhãn hoặc tính hợp lệ bất đồng hay chưa chắc chắn. Judge thứ ba không thấy phiếu hai judge trước. Mỗi judge trả nhãn ViHSD cho câu cuối và một đánh giá gộp về giữ nghĩa so với câu gốc cùng tuân thủ cấp độ của toàn chuỗi. Không đưa nhãn nguồn hoặc dự đoán classifier vào prompt. Code đối chiếu nhãn judge với nhãn nguồn để xác định `valid`: YES khi giữ nghĩa, đúng cấp độ và giữ nhãn; NO khi xác định ít nhất một điều kiện không đạt; các trường hợp còn lại là UNSURE. Phản hồi lỗi hoặc đầu vào quá dài được giữ UNSURE, không tự loại. Tổng hợp nhãn và valid bằng hai phiếu trùng nhau khác UNSURE; không đủ phiếu thì giữ chưa chắc chắn. Mọi mẫu YES/NO/UNSURE vẫn được giữ trong CSV để phân tích. Phiên bản hiện tại dùng kết quả tổng hợp LLM, chưa có phiếu human.

**Human review — hoãn sang phiên bản sau.** H3 và kiểm định gán tay chưa được thực hiện trong phiên bản hiện tại. Khi triển khai, dự kiến hai người gán độc lập và người thứ ba phân xử bất đồng, lưu riêng phiếu human để đối chiếu với LLM.

Trong phiên bản này, một cặp được chấp nhận khi bộ judge tổng hợp `valid=YES` cho đúng đường chỉnh sửa; việc qua bộ lọc tự động chưa đủ để xác nhận hợp lệ. Các ứng viên dùng để tuyên bố thành công hợp lệ phải được kiểm định; để kết luận một lần attack không có thành công hợp lệ, cần kiểm tra hết candidate pool tương ứng. Nếu nguồn lực hạn chế, xác định trước một tập đánh giá chính được kiểm định đầy đủ. Không loại mẫu chưa kiểm định khỏi mẫu số rồi diễn giải như ASR sau kiểm định LLM toàn test.

**Quy tắc đưa vào tập test đối kháng.** Giữ toàn bộ biến thể khác nhau vừa được bộ LLM judge chấp nhận vừa làm classifier bị tấn công dự đoán sai, với điều kiện câu gốc được classifier đó dự đoán đúng. Xét các ứng viên đã chấm trong ngân sách và cấu hình được khóa trước khi chạy test. Không giới hạn một biến thể mỗi câu gốc, không chọn riêng biến thể có margin lớn nhất và không cắt bớt số biến thể hợp lệ thành công theo câu gốc. Ví dụ, một câu gốc có 7 biến thể đáp ứng các điều kiện này thì giữ cả 7. Ứng viên chưa được kiểm định hoặc không đạt điều kiện được giữ trong nhật ký để phân tích, không đưa vào tập test đối kháng được LLM chấp nhận.

Hợp nhất kết quả từ các classifier và điều kiện attack đã định trước. Loại trùng theo cặp `(ID câu gốc, văn bản biến thể sau chuẩn hóa NFC)`; giữ đầy đủ nguồn sinh và kết quả kiểm định của từng điều kiện tạo ra cùng biến thể. Một biến thể chỉ cần có ít nhất một nguồn tấn công thành công hợp lệ để được giữ; việc hợp nhất không biến nó thành thành công của các điều kiện khác. Các biến thể khác nhau của cùng câu gốc vẫn được giữ riêng.

Trong bản phát hành cuối cùng, mỗi bản ghi dự kiến lưu ID gốc, nhãn, câu đối kháng, checkpoint/prompt/chế độ LLM, classifier/seed, lịch sử chỉnh sửa, số truy vấn, chi phí sinh, kết quả tuân thủ và annotation. Các biến thể của cùng câu giữ nguyên nhóm nguồn và split. Tập test đối kháng (challenge set) được phát hành kèm thông tin mô hình sinh và nhật ký thống kê thành công/thất bại; báo cáo cả số câu gốc duy nhất, tổng số biến thể và phân bố số biến thể mỗi câu gốc. Hai CSV pilot hiện tại được mô tả ở Mục 6.2; bước ba LLM judge đã chạy trên pilot; human review tạm hoãn và xuất tập test cuối cùng chưa được triển khai.

## 5. Experimental Setup

**Victim models.** Đã huấn luyện character TF-IDF + Linear SVM, PhoBERT-base [10], ViSoBERT [11] và XLM-R-base [12] với seed 42. Ba encoder chạy đủ 10 epoch, chọn checkpoint bằng macro-F1 validation cao nhất rồi đánh giá test. Cấu hình hiện tại: AdamW, learning rate $2\times10^{-5}$, batch size 16, max length 256, weight decay 0.01 và gradient clipping 1.0. PhoBERT tách từ bằng Underthesea; ViSoBERT và XLM-R nhận văn bản gốc. SVM dùng character TF-IDF n-gram 2–5, min-df 1, LinearSVC với C=1 và class-weight balanced. Các thiết lập khác dùng mặc định thư viện. Mục tiêu đánh giá cuối cùng là ba seed cho các mô hình có tính ngẫu nhiên; hiện mới hoàn thành một seed. Toàn bộ tiền xử lý được giữ nhất quán trước/sau attack.

**Comparisons.** Thí nghiệm chính so sánh C/W/S/MIX. Baseline sinh độc lập theo PromptAttack dùng cùng LLM và hướng dẫn cấp độ, sinh từ câu gốc mà không mở rộng qua beam; đây là bản thích nghi cho classifier độc lập, không đồng nhất với thiết lập LLM tự tấn công trong bài gốc. Đối chứng bổ sung gồm beam width $b=1$ và ba nhánh đơn cấp chạy riêng với cùng tổng $Q$ và $G$ nhưng không nối tiếp chỉnh sửa. Đối chứng cuối giúp phân biệt lợi ích thử nhiều chế độ với lợi ích phối hợp trong một chuỗi.

**Budgets and reproducibility.** Lượt chạy lại khảo sát trên validation với $Q=1,2,\ldots,25$, $G=5,10,\ldots,25$ và $b\in\{1,3,5\}$. Các mốc Q=10/50/100 và b=5 có tiền lệ ở TREPAT; giới hạn Q/G=25, dải ngân sách và b=1/3 là lựa chọn triển khai nhằm giảm chi phí. Với mỗi câu/chế độ/b, chỉ chạy một quỹ đạo đến tối đa Q=25 và G=25 rồi đọc kết quả ở mọi cặp Q/G. Có 125 snapshot Q/G trong bộ nhớ mỗi lượt tìm kiếm, sau đó tổng hợp qua các câu gốc thành 1.500 dòng/model trong `results.csv`; không tăng số lượt sinh hoặc truy vấn so với chỉ ghi mốc lớn nhất.

Vị trí sinh được trừ theo số lượng yêu cầu ở mỗi lượt gọi, kể cả đầu ra thiếu, lỗi, trùng hoặc bị lọc. Mỗi lượt yêu cầu tối đa 5 ứng viên nên G=25 cho phép tối đa 5 yêu cầu sinh, gồm cả yêu cầu được đáp ứng từ cache. Các mốc G trung gian là bội số của số ứng viên/lượt để không làm thay đổi prompt và kích thước lượt sinh; chỉ G lớn nhất được có lượt cuối yêu cầu phần ngân sách còn lại ít hơn 5. Đầu ra dư được ghi nhận nhưng không chấm. Prompt, seed từng lượt và thứ tự sinh/chấm không phụ thuộc các mốc Q/G nhỏ hơn. Chọn cấu hình chung trên validation, công bố cùng checkpoint LLM, prompt, seed và quy tắc dừng. Giữ chung giới hạn token đầu ra mỗi lượt gọi; báo cáo lượt gọi LLM, thời gian, số đầu ra bị loại và chi phí sinh lại. Loại trùng không tiêu tốn query mới; chi phí bộ lọc hiện nằm trong thời gian tìm kiếm tổng. Không điều chỉnh phương pháp theo kết quả test.

**Cache sinh ứng viên.** Bộ sinh giữ phản hồi thô trong RAM cho câu gốc đang xử lý. Chỉ dùng lại khi system/user prompt, seed và các tham số giải mã trùng nhau; model/tokenizer được cố định trong cùng instance. Cache giữ cả đầu ra lỗi hoặc rỗng, parse lại thành dữ liệu mới mỗi lần trả về và được xóa khi chuyển sang câu gốc khác; không lưu cache ra đĩa hoặc dùng chung giữa các phiên notebook. Mỗi cache hit vẫn tiêu tốn vị trí sinh G như một yêu cầu mới và không làm thay đổi cách tính Q. Seed dựa trên thứ tự yêu cầu, gồm cả cache hit. Giữ thứ tự beam 1/3/5; beam chạy trước tạo cache cho các yêu cầu trùng của beam sau. Chi phí thực tế và thời gian có thể phụ thuộc thứ tự/cache đã có; công bố chính sách cache và dùng tổng yêu cầu sinh khi so sánh công sức tìm kiếm giữa các cấu hình.

**Cấu hình chạy lại trên validation.** Dùng Qwen3-14B NF4 4-bit trên `cuda:0`, encoder classifier trên `cuda:1`; SVM dùng CPU. Notebook bật `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` trước khi khởi động tiến trình attack để giảm phân mảnh khi nạp. Dùng lại môi trường `.venv` với dependency chung, khóa bằng `uv.lock`; model dense không cần môi trường Transformers 4 riêng cho expert MoE. Mặc định seed 42, tối đa 2.048 token đầu ra/lượt. Bốn notebook `scripts/attack_validation_{svm,phobert,visobert,xlmr}.ipynb` nạp checkpoint classifier từ Kaggle Dataset và chạy inference. Mỗi classifier lấy ngẫu nhiên 10 câu được dự đoán đúng mỗi nhãn (30 câu/model), lấy mẫu cố định bằng seed 42. Trong từng classifier, cùng tập câu được dùng cho mọi chế độ và beam width; ID giữa các classifier có thể khác nhau do điều kiện dự đoán đúng. Có 360 lượt tìm kiếm/model (30 câu × 4 chế độ × 3 beam width); chạy một notebook tại một thời điểm. Đây là chạy lại trên validation, chưa phải đánh giá test. Ngưỡng cấu trúc C/W, số ứng viên, giới hạn ngân sách và số token đầu ra là lựa chọn triển khai; `max_span_tokens=4` chưa phải ngưỡng đã được xác lập cho tiếng Việt. Chưa có kết quả thực nghiệm của model 14B trong các bảng bên dưới. Sau lượt thử 30B thất bại, cần dọn cache trọng số cũ hoặc mở phiên Kaggle mới trước khi tải model mới.

Từng beam width chạy một lần với Q tối đa 25 và G=25; seed và thứ tự sinh/chấm không phụ thuộc ngân sách. Mọi cặp Q/G được lấy từ prefix của cùng lượt chạy; không chọn riêng các lần chạy có kết quả thuận lợi. Nếu hết vị trí sinh hoặc không còn ứng viên mới trước một mốc, giữ kết quả và số query thực dùng. Khi chọn cấu hình, xét ASR sau kiểm định cùng chi phí; không chọn Q chỉ vì ASR tự động cao nhất. Thời gian toàn quỹ đạo chỉ ghi một lần ở dòng Q=25/G=25 (cột `seconds`); các mốc nhỏ hơn để trống. Số lần model thực sự sinh (`llm_calls`) và lượt dùng lại cache (`cache_hits`) được ghi theo từng prefix Q/G, gồm cả phản hồi lỗi trước khi prefix kết thúc; tổng hai cột là số yêu cầu sinh; console chỉ tóm tắt cặp Q/G lớn nhất. Không lưu thống kê token riêng trong lượt chạy attack.

**Transfer setting.** Sinh trên mô hình A, khóa câu đối kháng rồi đánh giá trên mô hình B không tham gia tìm kiếm. Báo cáo khả năng chuyển giao cùng số lượng và thành phần nhãn của tập đủ điều kiện.

## 6. Preliminary Results and Implementation Status

### 6.1. Clean Baselines

Các giá trị dưới đây là phần trăm, lấy trực tiếp từ `metrics.csv` của từng run; validation là epoch có macro-F1 cao nhất. Tất cả là dữ liệu gốc chưa bị attack, mỗi model một run seed 42. SVM không có epoch.

| Model | Epoch được chọn | Val macro-F1 | Val accuracy | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Linear SVM | — | 62.59 | 86.34 | 61.83 | 86.72 |
| PhoBERT-base | 3 | 67.46 | 87.84 | 64.10 | 87.53 |
| ViSoBERT | 1 | 68.33 | 87.24 | 66.67 | 87.46 |
| XLM-R-base | 5 | 66.22 | 85.67 | 64.90 | 85.72 |

ViSoBERT có validation macro-F1 cao nhất (68,33%), có thể được chạy trước trong pilot bốn classifier dựa trên validation. Test macro-F1 của run này là 66,67%, cao hơn SVM 4,84 điểm phần trăm; PhoBERT có test accuracy cao nhất (87,53%). Các chênh lệch này là quan sát từ một seed, chưa có khoảng tin cậy hoặc kiểm định ý nghĩa thống kê và chưa cho biết độ bền trước attack.

Nguồn số liệu là `artifacts/baselines/<model>/metrics.csv`, với `<model>` lần lượt là `svm`, `phobert`, `visobert` và `xlmr`. Checkpoint tương ứng ở `artifacts/baselines/<model>/checkpoint/`. Mỗi model lưu trực tiếp `checkpoint/` và `metrics.csv`, không thêm thư mục seed hoặc thời gian. CSV lưu cả 10 epoch validation của encoder cùng kết quả test của checkpoint được chọn. Chạy lại model sẽ thay checkpoint và metrics tại cùng đường dẫn; muốn giữ một thí nghiệm khác có thể chỉ định `artifact_dir` riêng. Artifacts là dữ liệu cục bộ bị Git bỏ qua; bảng kết quả trong proposal và README được theo dõi cùng source.

### 6.2. Implementation and Next Experiments

Đã triển khai bốn notebook `scripts/attack_validation_{svm,phobert,visobert,xlmr}.ipynb`, nạp checkpoint baseline, lọc câu validation dự đoán đúng, sinh/lọc ứng viên C/W/S/MIX, beam search và grid dùng chung quỹ đạo query. Cấu hình baseline ở `src/vihsd_attack/classifiers/config.py`, cấu hình attack ở `src/vihsd_attack/attacks/config.py`; nhãn và thư mục artifacts dùng chung ở `src/vihsd_attack/config.py`, prompt ở `src/vihsd_attack/constants.py`. Notebook clone source và chạy qua môi trường Python 3.12 được khóa bằng `uv.lock`.

Pipeline lưu hai CSV đọc độc lập. `artifacts/attacks/<model>/candidates.csv` có 12 cột: `source_id`, `original`, `original_label`, `parent`, `candidate`, `predicted_label`, `mode`, `step_mode`, `beam_width`, `query`, `generation_used`, `status`. Mỗi dòng giữ ứng viên hoặc vị trí đầu ra lỗi/thiếu; có sẵn câu gốc, nhãn chuẩn và nội dung câu cha để đọc trực tiếp. `query` là thứ tự chấm classifier trong từng câu gốc/chế độ/beam; để trống cùng `predicted_label` nếu không chấm. `generation_used` là tổng vị trí sinh đã yêu cầu tại lần gọi sinh ứng viên, kể cả vị trí bị loại, thiếu hoặc lấy từ cache; các ứng viên cùng lần gọi có cùng giá trị. Để dựng pool tại Q/G, chọn đúng câu gốc/chế độ/beam, `status=scored`, `query <= Q` và `generation_used <= G`. Nội dung `parent` liên kết tới câu gốc hoặc ứng viên đã chấm trước đó trong cùng lượt tìm kiếm, phục hồi được chuỗi chỉnh sửa. Giữ các trạng thái `scored`, `format`, `mode`, `duplicate`, `missing`, `invalid_output`, `extra`, `unscored` để rà bộ lọc. JSON array rỗng hợp lệ sinh các vị trí `missing`; phản hồi rỗng, lỗi parse hoặc không phải array được ghi một dòng `invalid_output` giữ raw, các vị trí còn lại `missing`. Phần tử array sai kiểu/rỗng là `format`. Các vị trí bị loại không tiêu tốn query; phần tử dư `extra` không thêm Q/G.

`artifacts/attacks/<model>/results.csv` có 10 cột: `mode`, `beam_width`, `query_budget`, `generation_budget`, `samples`, `flips`, `queries`, `llm_calls`, `cache_hits`, `seconds`. Mỗi dòng tổng hợp các câu gốc đã chạy xong cho một cấu hình mode/beam/Q/G, kể cả thất bại. `flips` là số câu gốc có ít nhất một biến thể làm lật dự đoán, chưa xác nhận hợp lệ; `queries`, `llm_calls`, `cache_hits` là tổng chi phí trên các câu đó. Thời gian là tổng thời gian quỹ đạo đầy đủ, chỉ ghi ở Q/G lớn nhất; không cộng chi phí giữa các dòng ngân sách vì chúng dùng chung quỹ đạo. File được cập nhật sau mỗi lượt tìm kiếm, `samples` phản ánh mẫu số hiện tại. Grid chạy lại có 1.500 dòng/model, không lặp câu gốc và nhãn. Các CSV pilot trước dùng Q/G tối đa 100, có 24.000 dòng/model. Bốn bộ CSV pilot đã được chuyển schema mà không chạy lại model; vị trí sinh phục hồi từ ranh giới 5 vị trí/lần gọi và được đối chiếu với snapshot Q/G cũ. Margin chỉ dùng trong bộ nhớ, có thể chấm lại bằng checkpoint. Không thêm thư mục seed/thời gian; mỗi notebook lưu vào `artifacts/attacks/<model>/`, chạy lại thay hai CSV tại cùng đường dẫn. Gọi `run_pilot()` không có `output_dir` vẫn mặc định ở `artifacts/attacks/`. Đã có kết quả pilot Qwen3-8B có cache của cả bốn classifier và phiếu LLM judge theo Mục 6.3; chưa có human annotation.

Bước tiếp theo là chạy lại attack validation bằng Qwen3-14B trên 10 câu/nhãn/classifier, kiểm định các biến thể làm lật nhãn từ lượt chạy mới, rồi cố định một cấu hình chung cho C/W/S/MIX trước khi đánh giá test. Phiếu judge cũ chỉ áp dụng cho đường chỉnh sửa cũ, không tự coi là phiếu cho dữ liệu chạy lại. Human review và nâng cấp judge tạm hoãn. Đối chứng sinh độc lập, ba nhánh riêng, transfer và đánh giá nhiều seed vẫn là thí nghiệm dự kiến.

### 6.3. LLM judge implementation

Đã thêm `judges/{config,constants,data,generator,service}.py` và `scripts/judge.ipynb`. Bước chuẩn bị đọc CSV của bốn classifier, chỉ lấy các biến thể làm lật nhãn và phục hồi toàn bộ đường chỉnh sửa qua nội dung `parent`, kể cả câu cha chưa làm lật nhãn. Với pilot Qwen3-8B trước, sau loại trùng có 1.570 cặp câu gốc–biến thể và 1.773 chuỗi khác nhau. Các chuỗi trùng nhau giữa model/beam được chấm một lần; nguồn attack vẫn ở `candidates.csv`; các chuỗi khác nhau đi đến cùng biến thể được chấm riêng để không đánh đồng tuân thủ chế độ.

Qwen3.5-9B và SEA-LION v3.5-8B chạy 8-bit trên hai T4 riêng biệt, sau đó giải phóng GPU và chạy Gemma 4 12B ở 4-bit NF4 cho các trường hợp cần phiếu thứ ba [18–20]. Dùng FP16, SDPA, tắt thinking và greedy decoding; mặc định tối đa 128 token đầu ra, bảo vệ đầu vào 8.192 token mà không cắt văn bản. Ba checkpoint đã hoàn thành lượt pilot trên Kaggle T4. Đây là cấu hình triển khai hiện tại, chưa được chứng minh tối ưu hoặc hiệu chuẩn bằng human.

`judge_inputs.csv` dùng đúng 5 cột: `source_id`, `original`, `original_label`, `candidate`, `path`. Không tạo ID riêng hoặc lưu JSON nguồn attack. `path` là chuỗi một dòng dạng `1. [W] câu trung gian -> 2. [S] biến thể cuối`; giữ số thứ tự, chế độ yêu cầu và nội dung từng bước. Ký tự xuống dòng/backslash trong bình luận được escape; mũi tên thuộc nội dung bình luận được viết `\->` để phân biệt với dấu phân cách bước. Cả bốn file `qwen.csv`, `sealion.csv`, `gemma.csv`, `suggestions.csv` có 7 cột: 5 cột đầu vào, thêm `candidate_label`, `valid`. `candidate_label` là nhãn của riêng judge trong ba phiếu và nhãn tổng hợp trong file cuối, không phải prediction classifier; `valid` nhận YES/NO/UNSURE. Ghép phiếu và tiếp tục chạy bằng cặp `source_id` + `path`, không phụ thuộc thứ tự dòng; kiểm tra nội dung nguồn/nhãn khi đọc phiếu đã lưu. Nguồn classifier/chế độ/beam/Q/G có thể phục hồi từ candidate CSV bằng câu gốc và toàn bộ đường chỉnh sửa. Không lưu riêng meaning/mode, status, reason, suggestion, revision/signature hoặc file metadata. Ghi từng dòng để tiếp tục sau gián đoạn; dùng thư mục đầu ra khác khi đổi model/prompt/cấu hình. Trong phiên bản hiện tại, `suggestions.csv` là kết quả tổng hợp LLM để tính $\mathrm{ASR}_{\mathrm{LLM}}$; human review tạm hoãn. Không diễn giải kết quả này là human-validated. Qwen judge và bộ sinh cùng họ model là một hạn chế khi diễn giải đồng thuận.

Kết quả pilot: Qwen và SEA-LION mỗi model chấm 1.773 đường; Gemma chấm 925 đường bất đồng/chưa rõ. Tổng hợp có 172 đường YES (9,70%), tương ứng 141 cặp câu gốc–biến thể, và 1.601 đường NO. Có 495 nhãn biến thể UNSURE do thiếu đa số nhãn; những đường này vẫn có valid=NO theo tổng hợp tính hợp lệ. SEA-LION trả NO cho toàn bộ mẫu; do đó mọi đường YES đều cần Qwen và Gemma cùng chấp nhận. Đây là kết quả LLM-judged hiện tại, không phải tỷ lệ chính xác của judge hay ASR theo câu gốc.


## 7. Evaluation and Planned Analysis

Với mỗi mô hình, $E_f$ là tập ID câu gốc được dự đoán đúng. Mọi điều kiện attack dùng cùng tập này; thất bại hoặc hết ngân sách vẫn nằm trong mẫu số. $A_i$ là candidate pool của câu $i$ trong đúng chế độ/beam/Q/G:

$$
a_i = \mathbf{1}\left[\exists x' \in A_i:\ f(x') \ne y_i \land V_{\mathrm{LLM}}(x_i,x',y_i) = 1\right].
$$

$$
\mathrm{ASR}_{\mathrm{LLM}} = \frac{\sum_{i \in E_f} a_i}{|E_f|}.
$$

$\mathbf{1}[\cdot]$ bằng 1 khi điều kiện đúng và bằng 0 khi sai; $|E_f|$ là số câu gốc đủ điều kiện.

Mỗi ID nguồn đóng góp tối đa một thành công cho mỗi điều kiện. $V_{\mathrm{LLM}}$ phải lấy theo đúng đường chỉnh sửa, không áp dụng một phiếu YES cho mọi đường dẫn tới cùng biến thể.

Báo cáo **ASR tự động** trước judge và **$\mathrm{ASR}_{\mathrm{LLM}}$** sau tổng hợp ba judge, kèm tỷ lệ đường/biến thể được chấp nhận. $\mathrm{ASR}_{\mathrm{LLM}}$ là ước lượng theo bộ judge hiện tại, chưa có human xác nhận. Schema kiểm định gộp không tách được lỗi giữ nghĩa và lỗi cấp độ. Báo cáo riêng tỷ lệ qua bộ lọc cấu trúc trên các vị trí sinh đã yêu cầu; vị trí thiếu hoặc đầu ra lỗi không qua lọc. Tỷ lệ từ chối, trùng lặp và mức đạt trên đầu ra parse được được báo cáo riêng. Chỉ số bổ sung gồm truy vấn classifier, lượt gọi LLM, thời gian, ASR theo nhãn và chuyển đổi `HATE/OFFENSIVE -> CLEAN`.

Tập test đối kháng giữ toàn bộ biến thể thành công hợp lệ theo Mục 4; ASR vẫn tính theo ID câu gốc và từng điều kiện attack, không đếm mỗi biến thể thành một lần attack thành công độc lập. Điểm trên tập hợp nhất được báo cáo theo classifier bị nhắm tới và khả năng chuyển giao; tập này được chọn theo lỗi của các classifier tham gia xây dựng, nên không dùng riêng điểm trên tập hợp nhất để suy luận thứ hạng robustness tổng quát.

Để đo hiệu năng toàn test trước/sau attack cho từng classifier và điều kiện, thay mỗi câu gốc có attack thành công hợp lệ bằng toàn bộ $n_i$ biến thể tương ứng, mỗi biến thể mang trọng số $1/n_i$; nếu không có thì giữ câu gốc với trọng số 1. Các câu sai sẵn ở bản gốc được giữ nguyên. Báo cáo macro-F1 và F1 theo lớp từ ma trận nhầm lẫn có trọng số, để mỗi ID câu gốc có tổng trọng số bằng 1 như ở bản clean. Cách tính điểm này không loại bớt biến thể khỏi dataset phát hành.

Phân tích dự kiến gồm: (1) ASR theo ngân sách và tỷ lệ tuân thủ chế độ cho H1; (2) MIX so với từng cấp độ và đối chứng ba nhánh riêng cho H2; (3) chênh lệch trước/sau LLM judge và bất đồng giữa các judge; H3 về human annotation được hoãn; (4) ma trận chuyển giao và lỗi LLM ngoài phạm vi hướng dẫn. So sánh giữa mô hình bổ sung kết quả trên tập giao các câu cùng dự đoán đúng, kèm độ bao phủ. Ứng viên sai chế độ không được chuyển sang nhóm khác để tính thành công của điều kiện đã yêu cầu.

Dùng paired bootstrap theo **ID câu gốc** để ước lượng khoảng tin cậy, báo cáo riêng biến thiên giữa seed. Các so sánh chính và ngưỡng chênh lệch thực tiễn được chốt trước test; điều chỉnh kiểm định nhiều so sánh khi cần. Sự thay đổi tách từ và phân mảnh subword được xem là phân tích thăm dò.

## 8. Conclusion

Đã hoàn thành bốn baseline seed 42 và pilot dùng Qwen3-8B 8-bit, ba chế độ sinh ứng viên và beam search. Code chạy lại validation chuyển sang Qwen3-14B NF4 4-bit, 10 câu/nhãn, Q/G tối đa 25 và beam 1/3/5; chưa có kết quả lượt chạy mới. Đã có kết quả pilot validation cho cả bốn classifier; biến thể làm lật nhãn được gom thành 1.773 đường chỉnh sửa để kiểm định. Đã có kết quả ba LLM judge trên validation; phiên bản hiện tại phân tích $\mathrm{ASR}_{\mathrm{LLM}}$ và dùng giao thức này khi cố định cấu hình rồi đánh giá test. Human review và nâng cấp judge tạm hoãn. Chưa có kết luận xác nhận H1–H2; H3 chưa được thực hiện.

## Limitations

Kết quả baseline hiện chỉ có một seed; chưa thể suy luận độ ổn định hoặc ý nghĩa thống kê. Bộ lọc W dùng diff trên đơn vị chữ/số, nhạy với căn chỉnh và có thể nhận lỗi ký tự hoặc viết lại quá rộng đối với bình luận ngắn. Bộ lọc S chỉ kiểm tra hình thức, có thể nhận thay từ hoặc lỗi chính tả; đánh giá cấp độ sau đó phụ thuộc LLM judge. Các cấp độ không được giả định tách biệt tuyệt đối; tỷ lệ qua bộ lọc và tỷ lệ được judge chấp nhận phải được báo cáo riêng. Bộ judge chưa được hiệu chuẩn bằng human; SEA-LION trả NO cho mọi mẫu trong lượt hiện tại, nên tỷ lệ đồng thuận không phản ánh trực tiếp độ chính xác. Kết luận bị giới hạn bởi ViHSD, các classifier và checkpoint LLM được chọn. Cùng LLM và beam search vẫn có thể khác nhau về khả năng tuân thủ, chất lượng và chi phí giữa các chế độ; kết quả không tự động khái quát sang LLM khác. Không đo khoảng cách tổng nên không thể kết luận các điều kiện có cùng mức thay đổi văn bản; lợi ích MIX cần được diễn giải cùng với lịch sử chỉnh sửa và kết quả LLM judge; kiểm định bởi con người để sau. Ranh giới giữa thay từ và paraphrase có trường hợp mơ hồ, cần công bố quy tắc phân loại và mức đồng thuận. Annotation có thể bất đồng ở bình luận thiếu ngữ cảnh. Không tìm được adversarial example không chứng minh robustness tuyệt đối; thiết lập chính chỉ xét attacker được xem điểm số.

## Ethics Statement

Khi triển khai human annotation ở phiên bản sau, cần thông báo về nội dung xúc phạm và cơ chế bỏ qua mẫu gây khó chịu. Chia sẻ dữ liệu dẫn xuất tuân theo điều kiện ViHSD; nếu không được phân phối lại văn bản, cung cấp ID, metadata và mã tái dựng cho người có quyền truy cập. Tài nguyên được xây dựng nhằm đánh giá điểm yếu và hỗ trợ cải thiện hệ thống phát hiện nội dung độc hại.

## References

1. Luu et al. 2021. [A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts](https://arxiv.org/abs/2103.11528).
2. Li et al. 2019. [TextBugger: Generating Adversarial Text Against Real-world Applications](https://www.ndss-symposium.org/ndss-paper/textbugger-generating-adversarial-text-against-real-world-applications/). NDSS.
3. Jin et al. 2020. [Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment](https://ojs.aaai.org/index.php/AAAI/article/view/6311). AAAI.
4. Li et al. 2020. [BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://aclanthology.org/2020.emnlp-main.500/). EMNLP.
5. Ni et al. 2024. [Fraud's Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process](https://doi.org/10.1109/TKDE.2024.3349708). TKDE.
6. Przybyła et al. 2025. [Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models](https://aclanthology.org/2025.emnlp-main.1405/). EMNLP.
7. Morris et al. 2020. [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://aclanthology.org/2020.emnlp-demos.16/). EMNLP, System Demonstrations.
8. Morris et al. 2020. [Reevaluating Adversarial Examples in Natural Language](https://aclanthology.org/2020.findings-emnlp.341/). Findings of EMNLP.
9. Lent. 2025. [Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right](https://arxiv.org/abs/2507.03473). Preprint.
10. Nguyen and Nguyen. 2020. [PhoBERT: Pre-trained Language Models for Vietnamese](https://aclanthology.org/2020.findings-emnlp.92/). Findings of EMNLP.
11. Nguyen et al. 2023. [ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing](https://aclanthology.org/2023.emnlp-main.315/). EMNLP.
12. Conneau et al. 2020. [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/). ACL.
13. Xu et al. 2024. [An LLM can Fool Itself: A Prompt-Based Adversarial Attack](https://openreview.net/pdf?id=VVgGbB9TNV). ICLR. [Mã nguồn PromptAttack](https://github.com/GodXuxilie/PromptAttack).

14. Fang et al. 2025. [Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors](https://aclanthology.org/2025.emnlp-main.433/). EMNLP.
15. Qwen Team. 2025. [Qwen3-14B model card và cấu hình non-thinking](https://huggingface.co/Qwen/Qwen3-14B); [Qwen3-8B dùng trong pilot trước](https://huggingface.co/Qwen/Qwen3-8B); [mô tả hỗ trợ đa ngôn ngữ Qwen3](https://qwenlm.github.io/blog/qwen3/). Tài liệu model chính thức.
16. Le et al. 2022. [Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense](https://aclanthology.org/2022.findings-acl.232/). Findings of ACL.
17. Pham et al. 2013. [Vietnamese Text Accent Restoration with Statistical Machine Translation](https://aclanthology.org/Y13-1044/). PACLIC; tài liệu ngôn ngữ học tính toán về nhập nhằng khi bỏ dấu, không phải paper attack.

18. Qwen Team. [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B). Checkpoint judge thứ nhất; khác checkpoint sinh attack (Qwen3-8B ở pilot trước, Qwen3-14B ở lượt chạy lại).
19. AI Singapore. [Llama-SEA-LION-v3.5-8B-R model card](https://huggingface.co/aisingapore/Llama-SEA-LION-v3.5-8B-R). Checkpoint judge thứ hai và cấu hình `thinking_mode`.
20. Google DeepMind. [Gemma 4 12B IT model card](https://huggingface.co/google/gemma-4-12B-it). Checkpoint judge thứ ba; lượng tử hóa NF4 trong triển khai là cấu hình cần kiểm chứng riêng.
